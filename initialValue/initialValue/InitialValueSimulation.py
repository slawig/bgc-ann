#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
import numpy as np
import os
import time

import initialValue.constants as InitialValue_Constants
from initialValue.InitialValueDatabase import InitialValue_Database
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.metos3d.Metos3d import Metos3d
import metos3dutil.petsc.petscfile as petsc
from metos3dutil.simulation.AbstractClassSimulation import AbstractClassSimulation


class InitialValueSimulation(AbstractClassSimulation):
    """
    Class for the simulation with different initial concentrations
    """

    def __init__(self, metos3dModel, parameterId=0, timestep=1, concentrationTyp='constant', concentrationNum=0):
        """
        Initializes the simulation for different initial concentrations

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation
        concentrationTyp : {'vector', 'constant'}, default: 'constant'
            Use constant initial concentration or an initial concentration
            defined with vectors
        concentrationNum : int, default: 0
            Index of the concentration

        Attributes
        ----------
        _concentrationTyp : {'vector', 'constant'}
            Use constant initial concentration or an initial concentration
            defined with vectors
        _differentTracer : int
            Number of different tracer
        _concentrationNum : int
            Index of the concentration
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(InitialValue_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
        assert type(concentrationNum) is int and 0 <= concentrationNum

        #Time
        startTime = time.time()

        #Initial concentration
        self._concentrationTyp = concentrationTyp
        self._differentTracer = len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel])
        self._concentrationNum = concentrationNum

        AbstractClassSimulation.__init__(self, metos3dModel, parameterId=parameterId, timestep=timestep)

        logging.info('***Initialization of InitialValueSimulation:***\nMetos3dModel: {:s}\nParameterId: {:d}\nTime step: {:d}dt\nInitial concentration: Type {:s}, differentTracer {:d}, concentrationNum {:d}'.format(self._metos3dModel, self._parameterId, self._timestep, self._concentrationTyp, self._differentTracer, self._concentrationNum))
        logging.info('***Time for initialization: {:.6f}s***\n\n'.format(time.time() - startTime))


    def _init_database(self):
        """
        Inits the database connection

        Notes
        -----
        Sets the variable _database
        """
        self._database = InitialValue_Database()


    def set_concentrationId(self, concentrationId=None):
        """
        Sets the id of the initial concentration

        Parameters
        ----------
        concentrationId : int or None, default: None
            Id of the initial concentration. If None, uses the id of the
            default initial concentration
        """
        assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId

        if concentrationId is None:
            self._concentrationId = self._database.get_concentrationId(self._concentrationTyp, self._differentTracer, self._concentrationNum)
        else:
            self._concentrationId = concentrationId

        self._set_simulationId()
        self._set_path()


    def set_differentTracer(self, differentTracer):
        """
        Sets the number of different tracer

        Parameters
        ----------
        differentTracer : int
            Number of the different tracer
        """
        assert type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])

        self._differentTracer = differentTracer
        self.set_concentrationId()


    def _set_path(self):
        """
        Sets the path to the simulation directory
        """
        self._path = os.path.join(InitialValue_Constants.PATH, 'Data', self._metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), InitialValue_Constants.PATH_CONCENTRATION_TYP[self._concentrationTyp], InitialValue_Constants.PATH_DIFFERENT_TRACER[self._differentTracer], 'InitialTracer_{:0>3d}'.format(self._concentrationNum))


    def existsLogfile(self):
        """
        Returns if the logfile exists

        Returns
        -------
        bool
            True, if the logfile exists
        """
        logfile = os.path.join(InitialValue_Constants.PATH, 'Data', 'Logs', 'Logfile', InitialValue_Constants.PATTERN_LOGFILE.format(self._metos3dModel, self._parameterId, self._concentrationTyp, self._concentrationNum, self._differentTracer, self._timestep))
        return os.path.exists(logfile) and os.path.isfile(logfile)


    def _startSimulation(self):
        """
        Starts the spin up simulation

        Notes
        -----
        Creates the directory of the simulation
        """
        os.makedirs(self._path, exist_ok=True)

        metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._years, nodes=self._nodes)
        metos3d.setTrajectoryParameter(trajectoryYear=self._trajectoryYear)

        #Set the initial concentration for the spin up
        if self._concentrationTyp == 'constant':
            metos3d.setInitialConcentration([float(c) for c in self._database.get_concentration(self._concentrationId)[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[self._metos3dModel]]])
        else:
            distribution, tracerDistribution = self._database.get_distribution(self._concentrationId)
            metos3d.setInputDir(os.path.join(InitialValue_Constants.PATH_INITIAL_TRACER, InitialValue_Constants.PATH_DISTRIBUTION.format(distribution), InitialValue_Constants.PATH_TRACER_DISTRIBUTION.format(tracerDistribution), InitialValue_Constants.PATH_DIFFERENT_TRACER.format(self._differentTracer)))
            metos3d.setInputTracerName([InitialValue_Constants.PATTERN_TRACER_INITIAL_CONCENTRATION.format(i, self._concentrationNum) for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]))])

        if self._spinupTolerance is not None:
            metos3d.setTolerance(self._spinupTolerance)

        #Run the spin up simulation
        metos3d.run()


    def _set_calculateNormReferenceSimulationParameter(self):
        """
        Returns parameter of the norm calculation

        Returns
        -------
        tuple
            The tuple contains
              - the simulationId of the simulation used as reference
                simulation and
              - path of the directory of the reference simulation
        """
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
        simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference)

        pathReferenceTracer = os.path.join(InitialValue_Constants.PATH, 'Data', self._metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), InitialValue_Constants.PATH_CONCENTRATION_TYP['constant'], InitialValue_Constants.PATH_DIFFERENT_TRACER[len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])], 'InitialTracer_{:0>3d}'.format(0))

        return (simulationIdReference, pathReferenceTracer)


    def _initialTracerConcentration(self):
        """
        Returns a vector with the initial tracer concentration

        Returns
        -------
        numpy.ndarray
            Numpy array with the initial tracer concentration
        """
        initialConcentration = self._database.get_concentration(self._concentrationId)[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[self._metos3dModel]]
        if self._concentrationTyp == 'constant':
            initialConcentration = np.array([float(c) for c in initialConcentration])
            tracerInitialConcentration = initialConcentration * np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        else:
            tracerInitialConcentration = np.empty(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
            i = 0
            for tracerName in initialConcentration:
                tracerInitialConcentration[:,i] = petsc.readPetscFile(os.path.join(InitialValue_Constants.PATH_INITIAL_TRACER, InitialValue_Constants.PATH_DISTRIBUTION.format(distribution), InitialValue_Constants.PATH_TRACER_DISTRIBUTION.format(tracerDistribution), InitialValue_Constants.PATH_DIFFERENT_TRACER.format(self._differentTracer)))

        return tracerInitialConcentration

