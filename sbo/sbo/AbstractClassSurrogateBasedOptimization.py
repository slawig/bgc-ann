#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
import numpy as np
import os
import threading

from ann.network.FCN import FCN
from ann.network.ANN_SET_MLP import SET_MLP
from ann.geneticAlgorithm.geneticAlgorithm import GeneticAlgorithm
import metos3dutil.metos3d.Metos3d as Metos3d
import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants
import sbo.constants as SBO_Constants
from database.SBO_Database import SBO_Database


class AbstractClassSurrogateBasedOptimization(ABC):
    """
    Abstract class for a parameter optimization using a surrogate based optimization.
    @author: Markus Pfeil
    """

    def __init__(self, optimizationId, nodes=NeshCluster_Constants.DEFAULT_NODES, dbpath=SBO_Constants.DB_PATH):
        """
        Initialisation of the surrogate based optimization
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        assert type(nodes) is int and 0 < nodes

        #Logging
        self.queue = mp.Queue()
        self.logger = logging.getLogger(__name__)
        self.lp = threading.Thread(target=self.logger_thread)

        self._optimizationId = optimizationId
        self._nodes = nodes

        #Set the parameter of the optimization using the configuration saved in the database for the given optimizationId
        self._sboDB = SBO_Database(dbpath=dbpath)

        #Model
        self._setModel(self._sboDB.get_model(self._optimizationId))

        #Parameter of the high fidelity model
        (highFidelityModelYears, highFidelityModelTimestep) = self._sboDB.get_highFidelityModel(self._optimizationId)
        self._setHighFidelityModel(modelYears = highFidelityModelYears, timestep = highFidelityModelTimestep)

        #Parameter of the low fidelity model
        if self._sboDB.usesLowFidelityModelAnn(self._optimizationId):
            self._setLowFidelityModelAnn()
        else:
            #Parameter using a simulation with metos3d as low fidelity model
            (lowFidelityModelYears, lowFidelityModelTimestep) = self._sboDB.get_lowFidelityModel(self._optimizationId)
            self._setLowFidelityModel(modelYears = lowFidelityModelYears, timestep = lowFidelityModelTimestep)

        #Set parameter used to build the surrogate
        (enhancement, lowerBound, upperBound, threshold) = self._sboDB.get_surrogateCorrectionParameter(self._optimizationId)
        self._setSurrogateCorrectionParameter(enhancement = enhancement, lowerBound = lowerBound, upperBound = upperBound, threshold = threshold)

        self._setTrajectoryNorm(self._sboDB.get_trajectoryNorm(self._optimizationId))


    def close_connection(self):
        """
        Close the database connection
        @author: Markus Pfeil
        """
        self._sboDB.close_connection()


    def logger_thread(self):
        """
        Logging for multiprocessing
        @author: Markus Pfeil
        """
        while True:
            record = self.queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)


    def _setModel(self, model):
        """
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS

        self._model = model
        logging.info('****Set the model****Model: {}'.format(self._model))


    def _setHighFidelityModel(self, modelYears = 5000, timestep = 64):
        """
        Set the parameter of the high fidelity model
        @author: Markus Pfeil
        """
        assert type(modelYears) is int and 0 < modelYears
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        self._highFidelityModelYears = modelYears
        self._highFidelityModelTimestep = timestep
        logging.info('****Set parameter for the high fidelity model****\nModel years: {:d}\nTimestep: {:d}'.format(self._highFidelityModelYears, self._highFidelityModelTimestep))


    def _setLowFidelityModel(self, modelYears = 25, timestep = 64):
        """
        Set the parameter of the low fidelity model
        @author: Markus Pfeil
        """
        assert type(modelYears) is int and 0 < modelYears
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        self._lowFidelityModelUseAnn = False
        self._lowFidelityModelYears = modelYears
        self._lowFidelityModelTimestep = timestep
        logging.info('****Set parameter for the low fidelity model****\nUse ANN: {}\nModel years: {:d}\nTimestep: {:d}'.format(self._lowFidelityModelUseAnn, self._lowFidelityModelYears, self._lowFidelityModelTimestep))


    def _setLowFidelityModelAnn(self):
        """
        Set the parameter of the low fidelity model using an artificial neural network
        @author: Markus Pfeil
        """
        #Parameter using an artificial neural network as low fidelity model
        (annType, annNumber) = self._sboDB.get_annConfigLowFidelityModel(self._optimizationId)

        if annType == 'fcn':
            self._lowFidelityModelAnn = FCN(annNumber)
        elif annType == 'set':
            self._lowFidelityModelAnn = SET_MLP(annNumber)
        elif annType == 'setgen':
            geneticAlgorithm = GeneticAlgorithm(gid=annNumber)
            (gen, uid, ann_path) = geneticAlgorithm.readBestGenomeFile()
            self._lowFidelityModelAnn = SET_MLP(uid, setPath=False)
            self._lowFidelityModelAnn.set_path(ann_path)

        self._lowFidelityModelAnn.loadAnn()
        self._lowFidelityModelUseAnn = True
        self._lowFidelityModelUseCombinationAnnMetos3d = False

        #Use the prediction of the ann as initial concentration and run metos3d with this initial concentration
        (lowFidelityModelYears, lowFidelityModelTimestep) = self._sboDB.get_lowFidelityModel(self._optimizationId)
        if lowFidelityModelYears is not None and lowFidelityModelTimestep is not None:
            assert type(lowFidelityModelYears) is int and 0 < lowFidelityModelYears
            assert lowFidelityModelTimestep in Metos3d_Constants.METOS3D_TIMESTEPS

            self._lowFidelityModelYears = lowFidelityModelYears
            self._lowFidelityModelTimestep = lowFidelityModelTimestep
            self._lowFidelityModelUseCombinationAnnMetos3d = True
        else:
            self._lowFidelityModelTimestep = self._highFidelityModelTimestep

        logging.info('****Set parameter for the low fidelity model****\nUse ANN: {}\nANN type: {}\nANN number: {}{}'.format(self._lowFidelityModelUseAnn, annType, annNumber, '\nModel years: {:d}\nTimestep: {:d}'.format(self._lowFidelityModelYears, self._lowFidelityModelTimestep) if self._lowFidelityModelUseCombinationAnnMetos3d else ''))


    def _setSurrogateCorrectionParameter(self, enhancement = True, lowerBound = 0.1, upperBound = 5.0, threshold = 5 * 10**(-3)):
        """
        Set the parameter for computation of the surrogate correction vector
        @author: Markus Pfeil
        """
        assert type(enhancement) is bool
        assert (not enhancement and lowerBound is None) or (enhancement and type(lowerBound) is float and 0 < lowerBound)
        assert (not enhancement and upperBound is None) or (enhancement and type(upperBound) is float and lowerBound <= upperBound)
        assert (not enhancement and threshold is None) or (enhancement and type(threshold) is float and 0 < threshold)

        self._surrogateCorrectionEnhancement = enhancement
        self._surrogateCorrectionLowerBound = lowerBound
        self._surrogateCorrectionUpperBound = upperBound
        self._surrogateCorrectionThreshold = threshold
        logging.info('****Set parameter for the surrogate correction****\nEnhancement: {}\nLower bound: {:e}\nUpper bound: {:e}\nThreshold: {:e}'.format(self._surrogateCorrectionEnhancement, self._surrogateCorrectionLowerBound, self._surrogateCorrectionUpperBound, self._surrogateCorrectionThreshold))


    def _setTrajectoryNorm(self, useTrajectoryNorm):
        """
        Set the norm for the cost function (norm over the whole trajectory or norm only of the first point in time)
        @author: Markus Pfeil
        """
        assert type(useTrajectoryNorm) is bool

        self._useTrajectoryNorm = useTrajectoryNorm
        logging.info('****Set the trajectory norm to {}****'.format(self._useTrajectoryNorm))


    @abstractmethod
    def _get_path_lowFidelityModel(self):
        """
        Get the path of the low fidelity model for the current evaluation.
        @author: Markus Pfeil
        """
        pass


    def _lowFidelityModel(self, u, runMetos3d=True, remove=True):
        """
        Evaluation of the low fidelity model
        @author: Markus Pfeil
        """
        assert type(u) is list and len(u) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]
        assert type(runMetos3d) is bool
        assert type(remove) is bool

        logging.debug('***Model evaluation of the low fidelity model***\nModel parameter: {}'.format(u))
        y = np.zeros(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR/self._lowFidelityModelTimestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN)) if self._useTrajectoryNorm else np.zeros(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN))

        if self._lowFidelityModelUseAnn:
            #Use the prediction of the artificial neural network
            logging.debug('LowFidelityModel: Use the prediction of the ANN as approximation')
            tracerConcentration = self._lowFidelityModelAnn.predict(u)
            #Adjust the prediction to conserve mass
            tracerConcentration = self._adjustPrediction(tracerConcentration)

            if self._lowFidelityModelUseCombinationAnnMetos3d:
                #Run metos3d using the predicted tracer concentration as initial concentration
                logging.debug('LowFidelityModel: Calculate the approximation using Metos3d and the predicted concentration as initial concentration')
                metos3dSimulationPath = self._get_path_lowFidelityModel()
                os.makedirs(metos3dSimulationPath, exist_ok=not runMetos3d)
                lowFidelityModel = Metos3d.Metos3d(self._model, self._lowFidelityModelTimestep, u, metos3dSimulationPath, modelYears = self._lowFidelityModelYears, nodes = self._nodes)

                if self._useTrajectoryNorm:
                    lowFidelityModel.setCalculateTrajectory()
                    lowFidelityModel.setTrajectoryParameter(trajectoryYear=self._lowFidelityModelYears+1, trajectoryStep=1)

                if runMetos3d:
                    #Save prediction of the artificial neural network
                    i = 0
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                        lowFidelityModel.saveNumpyArrayAsPetscFile(Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer), tracerConcentration[:,i])
                        i = i + 1

                    lowFidelityModel.setInputDir(os.path.join(metos3dSimulationPath, 'Tracer'))
                    lowFidelityModel.run()

                #Read tracer concentration
                y = lowFidelityModel.readTracer()

                #Remove tracer of the trajectory
                if remove:
                    lowFidelityModel.removeTracer()
            elif self._useTrajectoryNorm:
                #Calculate trajectory
                logging.debug('LowFidelityModel: Calculate the trajectory using Metos3d')
                metos3dSimulationPath = self._get_path_lowFidelityModel()
                os.makedirs(metos3dSimulationPath, exist_ok=not runMetos3d)
                lowFidelityModel = Metos3d.Metos3d(self._model, self._lowFidelityModelTimestep, u, metos3dSimulationPath, modelYears = 0, nodes = self._nodes)

                #Run metos3d to calculate the trajectory
                lowFidelityModel.setCalculateOnlyTrajectory()

                if runMetos3d:
                    #Save prediction of the artificial neural network
                    i = 0
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                        lowFidelityModel.saveNumpyArrayAsPetscFile(Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer), tracerConcentration[:,i])
                        i = i + 1

                    lowFidelityModel.run()

                #Read tracer concentration
                y = lowFidelityModel.readTracer()

                #Remove tracer of the trajectory
                if remove:
                    lowFidelityModel.removeTracer()
            else:
                for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
                    y[i, :] = tracerConcentration[:,i]
        else:
            #Run metos3d
            logging.debug('LowFidelityModel: Calculate the approximation using Metos3d')
            metos3dSimulationPath = self._get_path_lowFidelityModel()
            os.makedirs(metos3dSimulationPath, exist_ok=not runMetos3d)
            lowFidelityModel = Metos3d.Metos3d(self._model, self._lowFidelityModelTimestep, u, metos3dSimulationPath, modelYears = self._lowFidelityModelYears, nodes = self._nodes)

            if self._useTrajectoryNorm:
                lowFidelityModel.setCalculateTrajectory()
                lowFidelityModel.setTrajectoryParameter(trajectoryYear=self._lowFidelityModelYears+1, trajectoryStep=1)

            if runMetos3d:
                lowFidelityModel.run()

            #Read tracer concentration
            y = lowFidelityModel.readTracer()

            #Remove tracer of the trajectory
            if remove:
                lowFidelityModel.removeTracer()

        return y


    def _adjustPrediction(self, tracerConcentration):
        """
        Adapt the mass of the tracer concentration to the overallMass in order to conserve mass.
        @author: Markus Pfeil
        """
        assert type(tracerConcentration) is np.ndarray
        assert np.shape(tracerConcentration) == (Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]))

        #Set negative concentration values to zero
        tracerConcentration = tracerConcentration.clip(min=0)

        #Volume of the boxes
        vol = Metos3d.readBoxVolumes()
        vol_vec = np.empty(shape=(len(vol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            vol_vec[:,i] = vol

        #Mass of the initial tracer concentration/prediction using the ANN
        overallMass = sum(Metos3d_Constants.INITIAL_CONCENTRATION[self._model]) * np.sum(vol_vec)
        massPrediction = np.sum(tracerConcentration * vol_vec)

        #Adjust the mass
        return (overallMass/massPrediction) * tracerConcentration


    @abstractmethod
    def _get_path_highFidelityModel(self):
        """
        Get the path of the high fidelity model for the current evaluation.
        @author: Markus Pfeil
        """
        pass


    def _highFidelityModel(self, u, runMetos3d=True, remove=True):
        """
        Evaluation of the high fidelity model.
        @author: Markus Pfeil
        """
        assert type(u) is list and len(u) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]
        assert type(runMetos3d) is bool
        assert type(remove) is bool

        logging.debug('***Model evaluation of the high fidelity model***\nModel parameter: {}'.format(u))
        #Run metos3d
        metos3dSimulationPath = self._get_path_highFidelityModel()
        os.makedirs(metos3dSimulationPath, exist_ok=not runMetos3d)
        highFidelityModel = Metos3d.Metos3d(self._model, self._highFidelityModelTimestep, u, metos3dSimulationPath, modelYears = self._highFidelityModelYears, nodes = self._nodes)

        if self._useTrajectoryNorm:
            highFidelityModel.setCalculateTrajectory()
            highFidelityModel.setTrajectoryParameter(trajectoryYear=self._highFidelityModelYears+1, trajectoryStep=1)

        if runMetos3d:
            highFidelityModel.run()

        #Read tracer concentration
        y = highFidelityModel.readTracer()

        #Remove tracer of the trajectory
        if remove:
            highFidelityModel.removeTracer()

        return y


    def _constructSurrogate(self, highFidelityTracer, lowFidelityTracer):
        """
        Construction of the surrogate
        @author: Markus Pfeil
        """
        assert type(highFidelityTracer) is np.ndarray
        assert type(lowFidelityTracer) is np.ndarray
        assert np.shape(highFidelityTracer) == np.shape(lowFidelityTracer)

        logging.debug('***Construct the surrogate***')
        self._surrogateCorrectionVector = highFidelityTracer / lowFidelityTracer

        if self._surrogateCorrectionEnhancement:
            self._surrogateCorrectionVector[self._surrogateCorrectionVector < self._surrogateCorrectionLowerBound] = self._surrogateCorrectionLowerBound
            self._surrogateCorrectionVector[self._surrogateCorrectionVector > self._surrogateCorrectionUpperBound] = self._surrogateCorrectionUpperBound

            self._surrogateCorrectionVector[highFidelityTracer < self._surrogateCorrectionThreshold] = 1.0
            self._surrogateCorrectionVector[lowFidelityTracer < self._surrogateCorrectionThreshold] = 1.0


    def _evaluateSurrogate(self, u, runMetos3d=True, remove=True):
        """
        Evaluation of the surrogate
        @author: Markus Pfeil
        """
        assert type(u) is list and len(u) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]
        assert type(runMetos3d) is bool
        assert type(remove) is bool

        logging.debug('***Evaluate the surrogate***\nModel parameter: {}'.format(u))
        lowFidelityTracer = self._lowFidelityModel(u, runMetos3d=runMetos3d, remove=remove)
        assert np.shape(lowFidelityTracer) == np.shape(self._surrogateCorrectionVector)
        return self._surrogateCorrectionVector * lowFidelityTracer

