#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import time
import logging
import numpy as np

import neshCluster.constants as NeshCluster_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.metos3d.Metos3d as Metos3d
import metos3dutil.petsc.petscfile as petsc
from ann.evaluation.AbstractClassEvaluation import AbstractClassEvaluation
import ann.network.constants as ANN_Constants
from ann.network.FCN import FCN
from ann.network.ANN_SET_MLP import SET_MLP
from ann.geneticAlgorithm.geneticAlgorithm import GeneticAlgorithm


class PredictionEvaluation(AbstractClassEvaluation):
    """
    Class to evaluate the prediction of an ANN.
    @author: Markus Pfeil
    """
    
    def __init__(self, annId, parameterId=0, years=1000, trajectoryYear=10, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2):
        """
        Constructor of the Prediction class
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1) 
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(years) is int and years > 0
        assert type(trajectoryYear) is int and trajectoryYear > 0
        assert type(massAdjustment) is bool
        assert tolerance is None or (type(tolerance) is float and tolerance > 0)
        assert type(spinupToleranceReference) is bool and ((not spinupToleranceReference) or (spinupToleranceReference and tolerance > 0))
        assert queue in NeshCluster_Constants.QUEUE
        assert type(cores) is int and cores > 0
        
        #Time
        self._startTime = time.time()
        
        AbstractClassEvaluation.__init__(self, annId, parameterId, massAdjustment=massAdjustment, tolerance=tolerance)
        
        #Model
        self._years = years
        logging.info('***Initialization of PredictionEvaluation:***\nANN: {}\nAnnId: {:d}\nModel: {}\nParameter id: {:d}\nSpin-up over {:d} years\nMass adjustment: {}'.format(self._annType, self._annId, self._model, self._parameterId, self._years, self._massAdjustment))
        
        #Parameter for metos3d
        self._getModelParameter()
        self._simulationPath = self._setSimulationPath()
        self._trajectoryYear = trajectoryYear
        self._lastSpinupYear = self._years + 1
        
        #Cluster parameter
        self._queue = queue
        self._cores = cores
        
        logging.info('***Time for initialisation: {:.6f}s***\n\n'.format(time.time() - self._startTime))


    def run(self, remove=False):
        """
        Start the simulation for the given model, parameterId and ann type.
        @author: Markus Pfeil
        """
        timeStart = time.time()
        if not self._spinupToleranceReference:
            self._generateInitalTracerConcentration()
        timeInitialTracer = time.time()
        self._startSimulation()
        timeSimulation = time.time()
        self._startOnestep(remove=remove)
        timeOnestep = time.time()
        logging.info('***Time for tracer initialisation {:.6f}s, simulation: {:.6f}s and time for onestep: {:.6f}s***\n\n'.format(timeInitialTracer - timeStart, timeSimulation - timeInitialTracer, timeOnestep - timeSimulation))


    def _generateInitalTracerConcentration(self):
        """
        Generate the tracer concentrations as initial concentration for the simulation with metos3d using the artificial neural network.
        @author: Markus Pfeil
        """
        tracerConcentration = None

        if self._annType == 'fcn':
            model = FCN(self._annNumber)
        elif self._annType == 'set':
            model = SET_MLP(self._annNumber)
        elif self._annType == 'setgen':
            geneticAlgorithm = GeneticAlgorithm(gid=self._annNumber)
            (gen, uid, ann_path) = geneticAlgorithm.readBestGenomeFile()
            model = SET_MLP(uid, setPath=False)
            model.set_path(ann_path)
            
        model.loadAnn()
        tracerConcentration = model.predict(self._parameterId)
        assert type(tracerConcentration) is np.ndarray and tracerConcentration.shape == (Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]))

        #Adapt the mass of the tracer concentration to the overallMass
        if self._massAdjustment:
            #Set negative concentration values to zero
            tracerConcentration = tracerConcentration.clip(min=0)

            #Volume of the boxes
            vol = Metos3d.readBoxVolumes()
            vol_vec = np.empty(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
            for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
                vol_vec[:,i] = vol

            #Mass of the initial tracer concentration/prediction using the ANN
            overallMass = sum(Metos3d_Constants.INITIAL_CONCENTRATION[self._model]) * np.sum(vol_vec)
            massPrediction = np.sum(tracerConcentration * vol_vec)

            #Adjust the mass
            tracerConcentration = (overallMass/massPrediction) * tracerConcentration

        i = 0
        for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
            tracerFilename = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_TRACER_INPUT.format(trac))
            petsc.writePetscFile(tracerFilename, tracerConcentration[:,i])
            i = i + 1


    def _startSimulation(self):
        """
        Start the simulation for the given model, parameterId and ann type.
        @author: Markus Pfeil
        """
        metos3d = Metos3d.Metos3d(self._model, self._timestep, self._modelParameter, self._simulationPath, modelYears = self._years, queue = self._queue, cores = self._cores)
        metos3d.setTrajectoryParameter(trajectoryYear=self._trajectoryYear)
        metos3d.setInputDir(self._simulationPath)
        if self._spinupTolerance:
            metos3d.setTolerance(self._tolerance)
        metos3d.run()
        
        if self._spinupTolerance:
            self._lastSpinupYear = metos3d.lastSpinupYear()


    def _startOnestep(self, remove=False):
        """
        Calculate the tracer concentration for the first time step of the next year.
        Metos3d generates the tracer concentration for the last time step in the year.
        @author: Markus Pfeil
        """
        assert type(remove) is bool

        metos3d = Metos3d.Metos3d(self._model, self._timestep, self._modelParameter, self._simulationPath, modelYears = self._years, queue = self._queue, cores = self._cores)
        for year in range(self._trajectoryYear, min(self._lastSpinupYear, self._years+1), self._trajectoryYear):
            metos3d.setOneStep(oneStepYear=year)
            metos3d.run()
            if remove:
                metos3d.removeTracer(oneStepYear=year)
