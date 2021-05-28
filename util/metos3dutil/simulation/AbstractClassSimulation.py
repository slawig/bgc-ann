#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
import numpy as np
import os
import shutil
import sqlite3
import time
import threading

import metos3dutil.database.constants as DB_Constants
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.metos3d.Metos3d import Metos3d, readBoxVolumes
import metos3dutil.petsc.petscfile as petsc
import neshCluster.constants as NeshCluster_Constants


class AbstractClassSimulation(ABC):
    """
    Abstract class for the simulation using metos3d
    """

    def __init__(self, metos3dModel, parameterId=0, timestep=1):
        """
        Initializes the simulation for different time steps

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation

        Attributes
        ----------
        _database
            Database connection inherited from the module 
            metos3dutil.database.DatabaseMetos3d
        _overwrite : bool, default: True
            Flag for the insert into the database. If True, overwrite existing
            database entries
        _metos3dModel : str
            Name of the biogeochemical model
        _parameterId : int
            Id of the parameter of the latin hypercube example
        _timestep : {1, 2, 4, 8, 16, 32, 64}
            Time step of the spin up simulation
        _concentrationId : int
            Id of the initial concentration
        _simulationId : int
            Id identifying the simulation in the datbase
        _modelParameter: list [float]
            List with the constant initial concentration
        _path : str
            Path of the simulation directory
        _years : int
            Model years of the spin up
        _trajectoryYear : int
            Interval saving the tracer concentration during the spin up
        _lastSpinupYear : int
            Number of model years of the spin up
        _spinupTolerance : float or None
            Tolerance of the spin up
        _trajectoryFlag : bool
            If True, calculate the trajectory for the evaluation
        _removeTracer : bool
            If True, remove the tracer after the one step calculation
        _nodes : int
            Number of nodes for the calculation on the high performance
            cluster
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        #Logging
        self.queue = mp.Queue()
        self.logger = logging.getLogger(__name__)
        self.lp = threading.Thread(target=self.logger_thread)

        #Database
        self._init_database()
        self.set_databaseOverwrite()

        #Metos3d model
        self._metos3dModel = metos3dModel
        self._parameterId = parameterId
        self._timestep = timestep
        self.set_concentrationId()
        self._getModelParameter()
        self.set_years()
        self.set_trajectoryYear()
        self.set_trajectoryFlag()
        self._lastSpinupYear = self._years + 1
        self._spinupTolerance = None
        self.set_removeTracer(removeTracer=False)
        self._set_simulationId()

        #Path
        self._set_path()

        #Cluster parameter
        self.set_nodes()


    def logger_thread(self):
        """
        Logging for multiprocessing
        """
        while True:
            record = self.queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)


    @abstractmethod
    def _init_database(self):
        """
        Inits the database connection

        Notes
        -----
        Sets the variable _database
        """
        pass


    def close_DB_connection(self):
        """
        Close the database connection
        """
        self._database.close_connection()


    def set_databaseOverwrite(self, overwrite=True):
        """
        Sets the database flag to overwrite entires

        Parameters
        ----------
        overwrite : bool, default: True
            If True, overwrite existing database entries
        """
        assert type(overwrite) is bool

        self._overwrite = overwrite


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
            self._concentrationId = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
        else:
            self._concentrationId = concentrationId


    def _getModelParameter(self):
        """
        Sets model parameter from the database
        """
        self._modelParameter = list(self._database.get_parameter(self._parameterId, self._metos3dModel))


    def set_years(self, years=10000):
        """
        Sets the model years of the spin up

        Parameters
        ----------
        years : int, default: 10000
            Model years of the spin up
        """
        assert type(years) is int and 0 < years

        self._years = years


    def set_trajectoryYear(self, trajectoryYear=50):
        """
        Sets the interval for saving tracer concentration during the spin up

        Parameters
        ----------
        trajectoryYear : int, default: 50
            Interval of model years to save the tracer concentration during
            the spin up simulation
        """
        assert type(trajectoryYear) is int and 0 < trajectoryYear

        self._trajectoryYear = trajectoryYear


    def set_trajectoryFlag(self, trajectoryFlag=True):
        """
        Sets the flag for evaluation using the trajectory

        Parameters
        ----------
        trajectoryFlag : bool, default: True
            If True, calculate the norm of the using the trajectory
        """
        assert type(trajectoryFlag) is bool

        self._trajectoryFlag = trajectoryFlag


    def set_removeTracer(self, removeTracer=True):
        """
        Sets the removeTracer flag

        Parameters
        ----------
        removeTracer : bool, default: True
            If True, remove tracer after one step calculation
        """
        assert type(removeTracer) is bool

        self._removeTracer = removeTracer


    def set_spinupTolerance(self, spinupTolerance=0.0001):
        """
        Sets the tolerance for the spin up

        Parameters
        ----------
        spinupTolerance : float, default: 0.0001
            Tolerance of the spin up
        """
        assert type(spinupTolerance) is float and 0 < spinupTolerance

        self._spinupTolerance = spinupTolerance


    def set_nodes(self, nodes=NeshCluster_Constants.DEFAULT_NODES):
        """
        Sets the number of nodes for the high performance cluster

        Parameters
        ----------
        nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
            Number of nodes on the high performance cluster
        """
        assert type(nodes) is int and 0 < nodes

        self._nodes = nodes


    @abstractmethod
    def _set_path(self):
        """
        Sets the path to the simulation directory
        """
        pass


    def _set_simulationId(self):
        """
        Sets the simulationId
        """
        self._simulationId = self._database.get_simulationId(self._metos3dModel, self._parameterId, self._concentrationId, timestep=self._timestep)


    def existsMetos3dOutput(self):
        """
        Returns if the output of metos3d exists

        Returns
        -------
        bool
            True, if the output exists for metos3d
        """
        metos3dOutput = os.path.join(self._path, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        return os.path.exists(metos3dOutput) and os.path.isfile(metos3dOutput)


    def run(self):
        """
        Runs the simulation

        Starts the spin up simulation and calculates the tracer concentration
        for the first time step of the model years using onestep simulation
        """
        timeStart = time.time()
        self._startSimulation()
        timeSimulation = time.time()
        self._startOnestep()
        timeOnestep = time.time()
        logging.info('***Time for simulation {:.6f}s and time for onestep {:.6f}s***\n'.format(timeSimulation - timeStart, timeOnestep - timeSimulation))


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
        metos3d.setInitialConcentration([float(c) for c in self._database.get_concentration(self._concentrationId)[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[self._metos3dModel]]])

        if self._spinupTolerance is not None:
            metos3d.setTolerance(self._spinupTolerance)

        #Run the spin up simulation
        metos3d.run()


    def _startOnestep(self):
        """
        Calculates the tracer concentration for the first time step

        Notes
        -----
        Metos3d generates the tracer concentration only for the last time step
        of a model year. This affects all model years for which the
        concentration is stored during the spin up. Therefore, a single time
        step is performed to obtain the tracer concentration for the first
        time instant of the model year, respectively.
        """
        metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears = self._years, nodes = self._nodes)
        for year in range(self._trajectoryYear, min(self._lastSpinupYear+1, self._years+1), self._trajectoryYear):
            metos3d.setOneStep(oneStepYear=year)
            metos3d.run()
            if self._removeTracer:
                metos3d.removeTracer(oneStepYear=year)


    def evaluation(self):
        """
        Evaluation of the spin up simulation

        Notes
        -----
        Inserts the values of the spin up norm, the norm of the tracer
        concentration as well as the norm of the concentration difference
        using a reference solution into the database. Moreover, the values of
        the norm over the whole trajectory is calculated and inserted for the
        last model year of the spin up.
        """
        #Insert the spin up values
        if self.existsMetos3dOutput() and not self._checkSpinupTotalityDatabase():
            self._insertSpinup()

        #Insert the tracer norm and deviation values
        if self.existsMetos3dOutput() and (not self._checkNormTotalityDatabase() or not self._checkDeviationTotalityDatabase()):
            self._calculateNorm()

        #Insert the norm of the trajectory
        if self._trajectoryFlag and self.existsMetos3dOutput() and not self._checkTrajectoryNormTotalityDatabase():
            self._calculateTrajectoryNorm()


    def _checkSpinupTotalityDatabase(self):
        """
        Checks, if the database contains all values of the spin up norm

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of spin up values
        """
        expectedCount = self._years

        if self._spinupTolerance is not None:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._spinupTolerance)
            if lastYear is not None:
                expectedCount = lastYear

        return self._database.check_spinup(self._simulationId, expectedCount)


    def _insertSpinup(self):
        """
        Inserts the spin up norm values into the database

        Reads the spin up norm values from the Metos3d job output and inserts
        these values into the database
        """
        metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._years, nodes=self._nodes)
        spinupNorm = metos3d.read_spinup_norm_values()
        spinupNormShape = np.shape(spinupNorm)

        try:
            for i in range(spinupNormShape[0]):
                year = int(spinupNorm[i,0])
                tolerance = float(spinupNorm[i,1])
                norm = float(spinupNorm[i,2]) if spinupNormShape[1] == 3 else None

                if year == 0 and tolerance == 0.0 and spinupNorm is not None and norm == 0.0:
                    raise ValueError()

                self._database.insert_spinup(self._simulationId, year, tolerance, norm, overwrite=self._overwrite)
        except (sqlite3.IntegrityError, ValueError):
            logging.error('Inadmissable values for simulationId {:0>4d} and year {:0>4d}\n'.format(self._simulationId, year))


    def _getTracerOutput(self, path, tracerPattern, year=None):
        """
        Retruns concentration values of all tracer for a given year

        Parameters
        ----------
        path : str
            Path to the directory including the tracer concentration files
        tracerPattern : str
            Pattern of the filenames of the tracer concentration
        year : int or None, default: None
            Model year of the spin up simulation

        Returns
        -------
        numpy.ndarray
            Numpy array with the tracer concentrations
        """
        assert os.path.exists(path) and os.path.isdir(path)
        assert type(tracerPattern) is str
        assert year is None or type(year) is int and year >= 0

        tracer_array = np.empty(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            if year is None:
                filename = tracerPattern.format(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel][i])
            else:
                filename = tracerPattern.format(year, Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel][i])
            tracerfile = os.path.join(path, filename)
            assert os.path.exists(tracerfile) and os.path.isfile(tracerfile)
            tracer = petsc.readPetscFile(tracerfile)
            assert len(tracer) == Metos3d_Constants.METOS3D_VECTOR_LEN
            tracer_array[:,i] = tracer
        return tracer_array


    @abstractmethod
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
        pass


    def _initialTracerConcentration(self):
        """
        Returns a vector with the initial tracer concentration

        Returns
        -------
        numpy.ndarray
            Numpy array with the initial tracer concentration
        """
        initialConcentration = self._database.get_concentration(self._concentrationId)[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[self._metos3dModel]]
        initialConcentration = np.array([float(c) for c in initialConcentration])
        return initialConcentration * np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))


    def _calculateNorm(self, simulationIdReference=None, pathReferenceTracer=None):
        """
        Calculates the tracer norm values for every tracer output

        Parameters
        ----------
        simulationIdReference : int or None, default: None
            Id of the simulation used as reference. If None, the function
            _set_calculateNormReferenceSimulationParameter is used to set this
            parameter.
        pathReferenceTracer : str or None, default: None
            Path of the reference simulation directory. If None, the function
            _set_calculateNormReferenceSimulationParameter is used to set this
            parameter.
        """
        assert simulationIdReference is None or type(simulationIdReference) is int and 0 <= simulationIdReference
        assert pathReferenceTracer is None or type(pathReferenceTracer) is str
        assert simulationIdReference is None and pathReferenceTracer is None or simulationIdReference is not None and pathReferenceTracer is not None

        #Parameter of the reference simulation
        if simulationIdReference is None or pathReferenceTracer is None:
            simulationIdReference, pathReferenceTracer = self._set_calculateNormReferenceSimulationParameter()
        pathReferenceTracer = os.path.join(pathReferenceTracer, 'Tracer')
        assert os.path.exists(pathReferenceTracer) and os.path.isdir(pathReferenceTracer)
        tracerReference = self._getTracerOutput(pathReferenceTracer, Metos3d_Constants.PATTERN_TRACER_OUTPUT, year=None)
        yearReference = 10000

        #Tracer's directories
        pathMetos3dTracer = os.path.join(self._path, 'Tracer')
        pathMetos3dTracerOneStep = os.path.join(self._path, 'TracerOnestep')
        assert os.path.exists(pathMetos3dTracer) and os.path.isdir(pathMetos3dTracer)
        assert os.path.exists(pathMetos3dTracerOneStep) and os.path.isdir(pathMetos3dTracerOneStep)

        #Read box volumes
        normvol = readBoxVolumes(normvol=True)
        vol = readBoxVolumes()
        euclidean = np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN))

        normvol_vec = np.empty(shape=(len(normvol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        vol_vec = np.empty(shape=(len(vol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        euclidean_vec = np.empty(shape=(len(euclidean), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            normvol_vec[:,i] = normvol
            vol_vec[:,i] = vol
            euclidean_vec[:,i] = euclidean
        normWeight = {'2': euclidean_vec, 'Boxweighted': normvol_vec, 'BoxweightedVol': vol_vec}

        #Insert the tracer norm of the metos3d calculation
        #Initial tracer concentration
        tracerInitialConcentration = self._initialTracerConcentration()

        for norm in DB_Constants.NORM:
            #Insert the norm values
            self._calculateTracerNorm(tracerInitialConcentration, 0, norm, normWeight[norm])
            self._calculateTracerDifferenceNorm(0, simulationIdReference, yearReference, tracerInitialConcentration, tracerReference, norm, normWeight[norm])

        self._calculateTracerDeviation(tracerInitialConcentration, 0)
        self._calculateTracerDifferenceDeviation(0, simulationIdReference, yearReference, tracerInitialConcentration, tracerReference)

        #Tracer concentrations during the spin up
        lastYear = self._years
        if self._spinupTolerance is not None:
            metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears = self._years, nodes = self._nodes)
            lastYear = metos3d.lastSpinupYear()

        for year in range(self._trajectoryYear, lastYear, self._trajectoryYear):
            tracerMetos3dYear = self._getTracerOutput(pathMetos3dTracerOneStep, Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR, year=year)

            for norm in DB_Constants.NORM:
                #Insert the norm values
                self._calculateTracerNorm(tracerMetos3dYear, year, norm, normWeight[norm])
                self._calculateTracerDifferenceNorm(year, simulationIdReference, yearReference, tracerMetos3dYear, tracerReference, norm, normWeight[norm])

            self._calculateTracerDeviation(tracerMetos3dYear, year)
            self._calculateTracerDifferenceDeviation(year, simulationIdReference, yearReference, tracerMetos3dYear, tracerReference)

        #Tracer concentration output of the spin up
        tracerMetos3d = self._getTracerOutput(pathMetos3dTracer, Metos3d_Constants.PATTERN_TRACER_OUTPUT, year=None)
        for norm in DB_Constants.NORM:
            #Insert the norm values
            self._calculateTracerNorm(tracerMetos3d, lastYear, norm, normWeight[norm])
            self._calculateTracerDifferenceNorm(lastYear, simulationIdReference, yearReference, tracerMetos3d, tracerReference, norm, normWeight[norm])

        self._calculateTracerDeviation(tracerMetos3d, lastYear)
        self._calculateTracerDifferenceDeviation(lastYear, simulationIdReference, yearReference, tracerMetos3d, tracerReference)


    def _checkNormTotalityDatabase(self):
        """
        Checks, if the database contains all values of the tracer norm

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of tracer norm values
        """
        years = self._years
        if self._spinupTolerance is not None:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._spinupTolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear

        checkNorm = True

        for norm in DB_Constants.NORM:
            checkNorm = checkNorm and self._database.check_tracer_norm(self._simulationId, expectedCount, norm=norm)

            #Norm of the differences
            concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
            simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference)
            checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, simulationIdReference, expectedCount, norm=norm)

        return checkNorm


    def _calculateTracerNorm(self, tracer, year, norm, normWeight):
        """
        Calculates the norm of a tracer concentration

        Parameters
        ----------
        tracer : numpy.ndarray
            Numpy array with the tracer concentration
        year : int
            Model year for the tracer concentraiton
        norm : str
            Type of the norm
        normWeight : numpy.ndarray
            Numpy array with the weights of the norm

        Notes
        -----
        Inserts the calculated norm into the database
        """
        assert type(tracer) is np.ndarray
        assert type(year) is int and year >= 0
        assert norm in DB_Constants.NORM
        assert type(normWeight) is np.ndarray

        tracerNorm = float(np.sqrt(np.sum((tracer)**2 * normWeight)))
        tracerSingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            tracerSingleValue[t] = float(np.sqrt(np.sum((tracer[:,t])**2 * normWeight[:,t])))

        self._database.insert_tracer_norm_tuple(self._simulationId, year, tracerNorm, tracerSingleValue[0], DOP=tracerSingleValue[1], P=tracerSingleValue[2], Z=tracerSingleValue[3], D=tracerSingleValue[4], norm=norm, overwrite=self._overwrite)


    def _calculateTracerDifferenceNorm(self, yearA, simulationId, yearB, tracerA, tracerB, norm, normWeight):
        """
        Calculates the norm of the difference of two tracers concentrations

        Parameters
        ----------
        yearA : int
            Model year for the first tracer concentration
        simulationId : int
            SimulationId of the second spin up simulation for the second
            tracer
        yearB : int
            Model year for the second tracer concentration
        tracerA : numpy.ndarray
            Numpy array with the first tracer concentration
        tracerB : numpy.ndarray
            Numpy array with the second tracer concentration
        norm : str
            Type of the norm
        normWeight : numpy.ndarray
            Numpy array with the weights of the norm

        Notes
        -----
        Inserts the calculated norm into the database
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(yearA) is int and yearA >= 0
        assert type(yearB) is int and yearB >= 0
        assert type(tracerA) is np.ndarray
        assert type(tracerB) is np.ndarray
        assert norm in DB_Constants.NORM
        assert type(normWeight) is np.ndarray

        tracerDifferenceNorm = float(np.sqrt(np.sum((tracerA - tracerB)**2 * normWeight)))
        tracerSingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            tracerSingleValue[t] = float(np.sqrt(np.sum((tracerA[:,t] - tracerB[:,t])**2 * normWeight[:,t])))
        self._database.insert_difference_tracer_norm_tuple(self._simulationId, simulationId, yearA, yearB, tracerDifferenceNorm, tracerSingleValue[0], DOP=tracerSingleValue[1], P=tracerSingleValue[2], Z=tracerSingleValue[3], D=tracerSingleValue[4], norm=norm, overwrite=self._overwrite)


    def _checkDeviationTotalityDatabase(self):
        """
        Checks, if the database contains all values of the tracer deviation

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of tracer norm values
        """
        years = self._years
        if self._spinupTolerance is not None:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._spinupTolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear

        checkDeviation = True
        checkDeviation = checkDeviation and self._database.check_tracer_deviation(self._simulationId, expectedCount)

        #Norm of the differences
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
        simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference)
        checkDeviation = checkDeviation and self._database.check_difference_tracer_deviation(self._simulationId, simulationIdReference, expectedCount)

        return checkDeviation


    def _calculateTracerDeviation(self, tracer, year):
        """
        Calculates the deviation of a tracer concentration

        Parameters
        ----------
        tracer : numpy.ndarray
            Numpy array with the tracer concentration
        year : int
            Model year for the tracer concentraiton

        Notes
        -----
        Inserts the deviation values into the database
        """
        assert type(tracer) is np.ndarray
        assert year is None or type(year) is int and year >= 0

        #Calculation of the mean value for all tracers
        mean = np.mean(tracer, axis=0)
        assert len(mean) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        #Calculation of the variance value for all tracers
        var = np.var(tracer, axis=0)
        assert len(var) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        #Calculation of the minimal values for all tracers
        minimum = np.nanmin(tracer, axis=0)
        assert len(minimum) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        #Calculation of the maximal values for all tracers
        maximal = np.nanmax(tracer, axis=0)
        assert len(maximal) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])

        meanValue = [None, None, None, None, None]
        varValue = [None, None, None, None, None]
        minimumValue = [None, None, None, None, None]
        maximumValue = [None, None, None, None, None]
        negativeCountValue = [None, None, None, None, None]
        negativeSumValue = [None, None, None, None, None]
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            meanValue[i] = float(mean[i])
            varValue[i] = float(var[i])
            minimumValue[i] = float(minimum[i])
            maximumValue[i] = float(maximal[i])
            #Calculation of the count of boxes with negative concentrations
            negativeCountValue[i] = int(np.count_nonzero(tracer[tracer[:,i]<0, i]))
            #Calculation of the sum of all negative concentrations
            negativeSumValue[i] = float(np.sum(tracer[tracer[:,i]<0, i]))

        self._database.insert_deviation_tracer_tuple(self._simulationId, year, meanValue[0], varValue[0], minimumValue[0], maximumValue[0], negativeCountValue[0], negativeSumValue[0], DOP_mean=meanValue[1], DOP_var=varValue[1], DOP_min=minimumValue[1], DOP_max=maximumValue[1], DOP_negative_count=negativeCountValue[1], DOP_negative_sum=negativeSumValue[1], P_mean=meanValue[2], P_var=varValue[2], P_min=minimumValue[2], P_max=maximumValue[2], P_negative_count=negativeCountValue[2], P_negative_sum=negativeSumValue[2], Z_mean=meanValue[3], Z_var=varValue[3], Z_min=minimumValue[3], Z_max=maximumValue[3], Z_negative_count=negativeCountValue[3], Z_negative_sum=negativeSumValue[3], D_mean=meanValue[4], D_var=varValue[4], D_min=minimumValue[4], D_max=maximumValue[4], D_negative_count=negativeCountValue[4], D_negative_sum=negativeSumValue[4], overwrite=self._overwrite)


    def _calculateTracerDifferenceDeviation(self, yearA, simulationId, yearB, tracerA, tracerB):
        """
        Calculates the deviation of the difference of two tracers concentrations

        Parameters
        ----------
        yearA : int
            Model year for the first tracer concentration
        simulationId : int
            SimulationId of the second spin up simulation for the second
            tracer
        yearB : int
            Model year for the second tracer concentration
        tracerA : numpy.ndarray
            Numpy array with the first tracer concentration
        tracerB : numpy.ndarray
            Numpy array with the second tracer concentration

        Notes
        -----
        Inserts the deviation values into the database
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(yearA) is int and yearA >= 0
        assert type(yearB) is int and yearB >= 0
        assert type(tracerA) is np.ndarray
        assert type(tracerB) is np.ndarray

        tracer = np.fabs(tracerA - tracerB)

        #Calculation of the mean value for all tracers
        mean = np.mean(tracer, axis=0)
        assert len(mean) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        #Calculation of the variance value for all tracers
        var = np.var(tracer, axis=0)
        assert len(var) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        #Calculation of the minimal values for all tracers
        minimum = np.nanmin(tracer, axis=0)
        assert len(minimum) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        #Calculation of the maximal values for all tracers
        maximal = np.nanmax(tracer, axis=0)
        assert len(maximal) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])

        meanValue = [None, None, None, None, None]
        varValue = [None, None, None, None, None]
        minimumValue = [None, None, None, None, None]
        maximumValue = [None, None, None, None, None]
        negativeCountValue = [None, None, None, None, None]
        negativeSumValue = [None, None, None, None, None]
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            meanValue[i] = float(mean[i])
            varValue[i] = float(var[i])
            minimumValue[i] = float(minimum[i])
            maximumValue[i] = float(maximal[i])
            #Calculation of the count of boxes with negative concentrations
            negativeCountValue[i] = int(np.count_nonzero(tracer[tracer[:,i]<0, i]))
            #Calculation of the sum of all negative concentrations
            negativeSumValue[i] = float(np.sum(tracer[tracer[:,i]<0, i]))

        self._database.insert_difference_tracer_deviation_tuple(self._simulationId, simulationId, yearA, yearB, meanValue[0], varValue[0], minimumValue[0], maximumValue[0], negativeCountValue[0], negativeSumValue[0], DOP_mean=meanValue[1], DOP_var=varValue[1], DOP_min=minimumValue[1], DOP_max=maximumValue[1], DOP_negative_count=negativeCountValue[1], DOP_negative_sum=negativeSumValue[1], P_mean=meanValue[2], P_var=varValue[2], P_min=minimumValue[2], P_max=maximumValue[2], P_negative_count=negativeCountValue[2], P_negative_sum=negativeSumValue[2], Z_mean=meanValue[3], Z_var=varValue[3], Z_min=minimumValue[3], Z_max=maximumValue[3], Z_negative_count=negativeCountValue[3], Z_negative_sum=negativeSumValue[3], D_mean=meanValue[4], D_var=varValue[4], D_min=minimumValue[4], D_max=maximumValue[4], D_negative_count=negativeCountValue[4], D_negative_sum=negativeSumValue[4], overwrite=self._overwrite)


    def _checkTrajectoryNormTotalityDatabase(self):
        """
        Checks, if the database contains values of the tracer trajectory norm

        Returns
        -------
        bool
            True if the number of database entries coincides with the expected
            number of tracer norm values
        """
        expectedCount = 1

        checkNorm = True
        for norm in DB_Constants.NORM:
            checkNorm = checkNorm and self._database.check_tracer_norm(self._simulationId, expectedCount, norm=norm, trajectory='Trajectory')

            #Norm of the differences
            concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
            simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference)
            checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, simulationIdReference, expectedCount, norm=norm, trajectory='Trajectory')

        return checkNorm


    def _calculateTrajectoryNorm(self):
        """
        Calculates the trajectory norm
        """
        #Parameter of the reference simulation
        simulationIdReference, referenceTrajectoryPath = self._set_calculateNormReferenceSimulationParameter()
        referenceTrajectoryPath = os.path.join(referenceTrajectoryPath, 'Trajectory')
        os.makedirs(referenceTrajectoryPath, exist_ok=True)
        yearReference = 10000
        trajectoryReference = self._calculateTrajectory(referenceTrajectoryPath, year=yearReference, timestep=self._timestep)

        #Read box volumes
        normvol = readBoxVolumes(normvol=True)
        vol = readBoxVolumes()
        euclidean = np.ones(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN))

        normvol_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / self._timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        vol_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / self._timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        euclidean_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / self._timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        for t in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / self._timestep)):
            for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
                normvol_vec[t,i,:] = normvol
                vol_vec[t,i,:] = vol
                euclidean_vec[t,i,:] = euclidean
        normWeight = {'2': euclidean_vec, 'Boxweighted': normvol_vec, 'BoxweightedVol': vol_vec}

        #Insert trajectory norm values of the reference simulation
        for norm in DB_Constants.NORM:
            if not self._database.check_tracer_norm(simulationIdReference, 1, norm=norm, trajectory='Trajectory'):
                self._calculateTrajectoryTracerNorm(trajectoryReference, yearReference, norm, normWeight[norm], timestep=self._timestep, simulationId=simulationIdReference)

        #Trajectory of the simulation
        trajectoryPath = os.path.join(self._path, 'Trajectory')
        os.makedirs(trajectoryPath, exist_ok=True)

        lastYear = self._years
        if self._spinupTolerance is not None:
            metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._years, nodes=self._nodes)
            lastYear = metos3d.lastSpinupYear()

        #Read trajectory
        trajectory = self._calculateTrajectory(trajectoryPath, year=self._years, timestep=self._timestep)

        for norm in DB_Constants.NORM:
            self._calculateTrajectoryTracerNorm(trajectory, self._years, norm, normWeight[norm], timestep=self._timestep)
            self._calculateTrajectoryDifferenceTracerNorm(self._years, simulationIdReference, yearReference, trajectory, trajectoryReference, norm, normWeight[norm], timestep=self._timestep)

        #Remove the directory for the trajectory
        shutil.rmtree(trajectoryPath, ignore_errors=True)


    def _existsTrajectory(self, trajectoryPath, timestep=1):
        """
        Checks, if the trajectory exists

        Parameters
        ----------
        trajectoryPath : str
            Path of the trajectory directory
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation to calculate the trajectory

        Returns
        -------
        bool
            True if the tracer exists for the whole trajectory
        """
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        trajectoryExists = os.path.exists(trajectoryPath) and os.path.isdir(trajectoryPath)
        if trajectoryExists:
            for index in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                    trajectoryFilename = Metos3d_Constants.PATTERN_TRACER_TRAJECTORY.format(0, index, tracer)
                    trajectoryExists = trajectoryExists and os.path.exists(os.path.join(trajectoryPath, trajectoryFilename)) and os.path.isfile(os.path.join(trajectoryPath, trajectoryFilename))
        return trajectoryExists


    def _calculateTrajectoryTracerNorm(self, trajectory, year, norm, normWeight, timestep=1, simulationId=None):
        """
        Calculates the trajectory norm for a given trajectory

        Parameters
        ----------
        trajectory : numpy.ndarray
            Numpy array with the trajectory for all tracers
        year : int
            Model year of the spin up simulation for the trajectory
        norm : str
            Type of the norm
        normWeight : numpy.ndarray
            Numpy array with the weights of the norm
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation
        simulationId : int or None
            SimulationId of the spin up simulation for the trajectory.
            If None, use the attribute simulationId
        """
        assert type(trajectory) is np.ndarray
        assert type(year) is int and year >= 0
        assert norm in DB_Constants.NORM
        assert type(normWeight) is np.ndarray
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert simulationId is None or type(simulationId) is int and 0 <= simulationId

        dt = 1.0 / int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)
        trajectoryNorm = float(np.sqrt(np.sum(trajectory**2 * normWeight) * dt))
        trajectorySingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            trajectorySingleValue[t] = float(np.sqrt(np.sum((trajectory[:,t,:])**2 * normWeight[:,t,:]) * dt))

        simId = self._simulationId if simulationId is None else simulationId
        self._database.insert_tracer_norm_tuple(simId, year, trajectoryNorm, trajectorySingleValue[0], DOP=trajectorySingleValue[1], P=trajectorySingleValue[2], Z=trajectorySingleValue[3], D=trajectorySingleValue[4], norm=norm, trajectory='Trajectory', overwrite=self._overwrite)


    def _calculateTrajectoryDifferenceTracerNorm(self, yearA, simulationId, yearB, trajectoryA, trajectoryB, norm, normWeight, timestep=1):
        """
        Calculates the trajectory norm for the difference of two trajectories

        Parameters
        ----------
        yearA : int
            Model year of the spin up simulation for the first trajectory
        simulationId : int
            SimulationId of the second spin up simulation for the second
            tracer
        yearB : int
            Model year of the spin up simulation for the second trajectory
        trajectoryA : numpy.ndarray
            Numpy array with the first trajectory for all tracers
        trajectoryB : numpy.ndarray
            Numpy array with the second trajectory for all tracers
        norm : str
            Type of the norm
        normWeight : numpy.ndarray
            Numpy array with the weights of the norm
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation to calculate the trajectory
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(yearA) is int and yearA >= 0
        assert type(yearB) is int and yearB >= 0
        assert type(trajectoryA) is np.ndarray
        assert type(trajectoryB) is np.ndarray
        assert norm in DB_Constants.NORM
        assert type(normWeight) is np.ndarray
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        dt = 1.0 / int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)
        trajectoryDifferenceNorm = float(np.sqrt(np.sum((trajectoryA - trajectoryB)**2 * normWeight) * dt))
        trajectorySingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            trajectorySingleValue[t] = float(np.sqrt(np.sum((trajectoryA[:,t,:] - trajectoryB[:,t,:])**2 * normWeight[:,t,:]) * dt))

        self._database.insert_difference_tracer_norm_tuple(self._simulationId, simulationId, yearA, yearB, trajectoryDifferenceNorm, trajectorySingleValue[0], DOP=trajectorySingleValue[1], P=trajectorySingleValue[2], Z=trajectorySingleValue[3], D=trajectorySingleValue[4], norm=norm, trajectory='Trajectory', overwrite=self._overwrite)


    def _calculateTrajectory(self, metos3dSimulationPath, year=10000, timestep=1, modelYears=0):
        """
        Calculates the trajectory

        Parameters
        ----------
        metos3dSimulationPath : str
            Path for the simulation with Metos3d
        year: int, default: 10000
            Model year of the spin up simulation for the trajectory
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation to calculate the trajectory
        modelYears : int
            Model years for Metos3d

        Returns
        -------
        numpy.ndarray
            Numpy array with the trajectory
        """
        assert os.path.exists(metos3dSimulationPath) and os.path.isdir(metos3dSimulationPath)
        assert year in range(0, self._years+1)
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(modelYears) is int and 0 <= modelYears

        #Run metos3d
        tracer_path = os.path.join(metos3dSimulationPath, 'Tracer')
        os.makedirs(tracer_path, exist_ok=True)

        metos3d = Metos3d(self._metos3dModel, timestep, self._modelParameter, metos3dSimulationPath, modelYears=modelYears, nodes=self._nodes)
        metos3d.setCalculateOnlyTrajectory()

        if not self._existsTrajectory(tracer_path, timestep=timestep):
            #Copy the input tracer for the trajectory
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                if year == 10000:
                    inputTracer = os.path.join(self._path, 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
                else:
                    inputTracer = os.path.join(self._path, 'TracerOnestep', Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer))
                shutil.copy(inputTracer, os.path.join(tracer_path, Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer)))

            metos3d.run()

        #Read tracer concentration
        trajectory = metos3d.readTracer()

        return trajectory

