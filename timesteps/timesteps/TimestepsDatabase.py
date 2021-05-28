#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
import numpy as np
import os
import sqlite3
import time

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.database.DatabaseMetos3d import DatabaseMetos3d
import timesteps.constants as Timesteps_Constants 



class Timesteps_Database(DatabaseMetos3d):
    """
    Access functions for the database
    """

    def __init__(self, dbpath=Timesteps_Constants.DB_PATH, completeTable=True, createDb=False):
        """
        Initialization of the database connection

        Parameter
        ----------
        dbpath : str, default: timesteps.constants.DB_PATH
            Path to the sqlite database
        completeTable : bool, default: True
            If the value is True, use all columns (even columns with value
            None) in SELECT queries on the database
        createDb : bool, default: False
            If True, the database does not have to exist and can be created
            using the function create_database

        Raises
        ------
        AssertionError
            If the file for the sqlite database does not exist
        """
        assert type(createDb) is bool
        assert createDb or os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        DatabaseMetos3d.__init__(self, dbpath, completeTable=completeTable, createDb=createDb)


    def create_database(self):
        """
        Create all tables of the database
        """
        self._c.execute('PRAGMA foreign_keys=on')
        self._create_table_parameter()
        self._create_table_initialConcentration()
        self._create_table_measurement()
        self._create_table_simulation()
        self._create_table_spinup()
        self._create_table_costfunctionEvaluation()
        self._create_table_convergence()

        #Database tables for the norm
        for norm in DB_Constants.NORM:
            self._create_table_tracerNorm(norm=norm)
            self._create_table_tracerNorm(norm=norm, trajectory='Trajectory')
            self._create_table_tracerDifferenceNorm(norm=norm)
            self._create_table_tracerDifferenceNorm(norm=norm, trajectory='Trajectory')

        #Database tables for the deviation
        self._create_table_DeviationTracer()
        self._create_table_DeviationTracerDifference()

        self._conn.commit()

        #Initial insert of data sets
        self._init_database()


    def _init_database(self):
        """
        Initial insert of the database tables

        Notes
        -----
        The functions inserts data sets into the tables Parameter,
        InitialConcentration, Measurement and Simulation
        """
        #Insert the reference parameter set and the parameter of the latin hypercube sample with 100 samples into the table Parameter
        self._init_table_parameter(referenceParameter=True, latinHypercubeSamples=(True, False, False))

        #Insert the constant initial concentration
        self._init_table_initialConcentration()

        #Insert the measurement
        self._init_table_measurement()

        #Insert the simulation data sets
        self._init_table_simulation()


    def _init_table_initialConcentration(self):
        """
        Initial insert of initial concentration data sets
        """
        purchases = []
        concentrationId = 0

        #Standard constant initial concentration
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS[:-1]:
            concentration = [Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][0]] + Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][1:-1]
            while len(concentration) < len(Metos3d_Constants.TRACER_MASK)-1:
                concentration.append(None)
            concentration.append(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][-1] if len(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel]) > 1 else None)

            purchases.append((concentrationId, 'constant') + tuple(concentration))
            concentrationId += 1

        #Same constant initial concentration for all tracer
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS[1:-1]:
            concentrationArray = (float(np.sum(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel])/(len(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel])))) * np.ones(len(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel]))
            concentration = list(concentrationArray[:-1])
            while len(concentration) < len(Metos3d_Constants.TRACER_MASK)-1:
                concentration.append(None)
            concentration.append(concentrationArray[-1] if len(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel]) > 1 else None)

            purchases.append((concentrationId, 'constant') + tuple(concentration))
            concentrationId += 1

        concentrationTuple = [(2.15, 0.0101, None, None, 0.0101), (2.07, 0.0501, None, None, 0.0501), (1.97, 0.1001, None, None, 0.1001), (1.77, 0.2001, None, None, 0.2001), (1.57, 0.3001, None, None, 0.3001), (1.37, 0.4001, None, None, 0.4001), (1.17, 0.5001, None, None, 0.5001), (0.97, 0.6001, None, None, 0.6001), (0.77, 0.7001, None, None, 0.7001), (2.14, 0.0101, 0.0101, None, 0.0101), (2.02, 0.0501, 0.0501, None, 0.0501), (1.87, 0.1001, 0.1001, None, 0.1001), (1.57, 0.2001, 0.2001, None, 0.2001), (1.27, 0.3001, 0.3001, None, 0.3001), (0.97, 0.4001, 0.4001, None, 0.4001), (0.67, 0.5001, 0.5001, None, 0.5001), (2.13, 0.0101, 0.0101, 0.0101, 0.0101), (1.97, 0.0501, 0.0501, 0.0501, 0.0501), (1.77, 0.1001, 0.1001, 0.1001, 0.1001), (1.37, 0.2001, 0.2001, 0.2001, 0.2001), (0.97, 0.3001, 0.3001, 0.3001, 0.3001), (0.57, 0.4001, 0.4001, 0.4001, 0.4001)]
        for concentration in concentrationTuple:
            purchases.append((concentrationId, 'constant') + concentration)
            concentrationId += 1

        self._c.executemany('INSERT INTO InitialConcentration VALUES (?,?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def _init_table_simulation(self):
        """
        Initial insert of simulation data sets
        """
        concentrationDic = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}

        purchases = []
        simulationId = 0

        for parameterId in range(Timesteps_Constants.PARAMETERID_MAX+1):
            for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
                    purchases.append((simulationId, metos3dModel, parameterId, concentrationDic[metos3dModel], timestep))
                    simulationId += 1

        self._c.executemany('INSERT INTO Simulation VALUES (?,?,?,?,?)', purchases)
        self._conn.commit()


    def get_simulationId(self, model, parameterId, concentrationId, timestep=1):
        """
        Returns the simulationId

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        timestep : int, default: 1
            Timestep used for the spin up calculation

        Returns
        -------
        int
            simulationId for the combination of model, parameterId,
            concentrationId and timestep

        Raises
        ------
        AssertionError
            If no entry for the model, parameterId, concentrationId and
            timestep exists in the database table Simulation
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and concentrationId >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND timestep = ?'
        self._c.execute(sqlcommand, (model, parameterId, concentrationId, timestep))
        simulationId = self._c.fetchall()

        assert len(simulationId) == 1
        return simulationId[0][0]


    def _create_table_measurement(self):
        """
        Create table Measurement
        """
        self._c.execute('''CREATE TABLE Measurement (measurementId INTEGER NOT NULL, tracer TEXT NOT NULL, PRIMARY KEY (measurementId))''')


    def _init_table_measurement(self):
        """
        Initial insert of measurement data sets
        """
        purchases = []
        measurementId = 0

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS[:-1]:
            tracer = '{:s}'.format(', '.join(map(str, Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel])))
            purchases.append((measurementId, tracer))
            measurementId += 1

        self._c.executemany('INSERT INTO Measurement VALUES (?,?)', purchases)
        self._conn.commit()


    def get_measurement_id(self, tracer):
        """
        Returns the measurementId for the list of tracers

        Parameters
        ----------
        tracer : list [str]
            List of tracer names

        Returns
        -------
        int
            measurementId for the given tracer names

        Raises
        ------
        AssertionError
            If no measurementId exists for the tracer names
        """
        assert type(tracer) is list

        sqlcommand = 'SELECT measurementId FROM Measurement WHERE tracer = ?'
        self._c.execute(sqlcommand, ('{}'.format(', '.join(map(str, tracer))),))
        measurement_id = self._c.fetchall()
        assert len(measurement_id) == 1
        return measurement_id[0][0]


    def _create_table_costfunctionEvaluation(self):
        """
        Create table CostfunctionEvaluation
        """
        self._c.execute('''CREATE TABLE CostfuctionEvaluation (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, measurementId INTEGER NOT NULL REFERENCES Measurement(measurementId), OLS REAL NOT NULL, WLS REAL NOT NULL, GLS REAL NOT NULL, PRIMARY KEY (simulationId, year, measurementId))''')


    def get_costfunction_evaluation_year(self, simulationdId, measurementId):
        """
        Returns the last year of the costfunction evaluation

        Returns the last year of the costfunction evaluation for the given
        simulationId and measurementId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        measurementId : int
            Id defining the tracers used in the cost function evaluation

        Returns
        -------
        int
            Last year for which an evaluation is available in the database
            If no evaluation is available, returns 0

        Raises
        ------
        AssertionError
            If more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(measurementId) is int and measurementId >= 0

        sqlcommand = 'SELECT c.year FROM CostfuctionEvaluation AS c WHERE c.simulationId = ? AND c.measurementId = ? AND NOT EXISTS (SELECT * FROM CostfuctionEvaluation AS d WHERE c.simulationId = d.simulationId AND c.measurementId = d.measurementId AND c.year < d.year)'
        self._c.execute(sqlcommand, (simulationdId, measurementId))
        yearList = self._c.fetchall()
        assert len(yearList) == 1 or len(yearList) == 0
        if len(yearList) == 0:
            year = 0
        else:
            year = yearList[0][0]
        return year


    def insert_costfunction_evaluation(self, simulationId, year, measurementId, ols, wls, gls, overwrite=False):
        """
        Insert cost function values

        Insert cost function values for one year of a simulation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year for the cost function evaluation
        measurementId : int
            Id defining the tracers used in the cost function evaluation
        ols : float
            Cost function value using ordinary least squares (OLS)
        wls : float
            Cost function value using weighted least squares (WLS)
        gls : float
            Cost function value using generalised least squares (GLS)
        overwrite : bool, default: False
            If True, remove existing value for the simulationId, year and
            measurementId before insert

        Raises
        ------
        AssertionError
            If an entry exists for the combination of simulationId, year and
            measurementId using overwriting=False
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(ols) is float
        assert type(wls) is not float
        assert type(gls) is not float
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId, year FROM CostfuctionEvaluation WHERE simulationId = ? AND year = ? AND measurementId = ?'
        self._c.execute(sqlcommand, (simulationId, year, measurementId))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand_delete = 'DELETE FROM CostfuctionEvaluation WHERE simulationId = ? AND year = ? AND measurementId = ?'
                self._c.execute(sqlcommand_delete, (simulationId, year, measurementId))
        else:
            assert len(dataset) == 0

        #Insert costfunction value
        purchases = []
        purchases.append((simulationId, year, measurementId, ols, wls, gls))
        self._c.executemany('INSERT INTO CostfuctionEvaluation VALUES (?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def _create_table_convergence(self):
        """
        Create table Convergence
        """
        self._c.execute('''CREATE TABLE Convergence (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), convergence INTEGER NOT NULL, PRIMARY KEY (simulationId))''')


    def check_convergence(self, simulationId):
        """
        Returns if en entries for the convergence exists

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        bool
           True if an database entry exists for the simulationId
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT convergence FROM Convergence WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        convergence = self._c.fetchall()
        return len(convergence) > 0


    def get_convergence(self, simulationId):
        """
        Returns the convergence behaviour

        Returns the convergence behaviour for the spin up calculation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        
        Returns
        -------
        bool
            True in the case of a convergent spin up calculation otherwise
            False

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT convergence FROM Convergence WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        convergence = self._c.fetchall()
        assert len(convergence) == 1
        return bool(convergence[0][0])


    def insert_convergence(self, simulationId, convergence, overwrite=False):
        """
        Insert the convergence behaviour

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        convergence : bool
            True for a convergent spin up calculation otherwise False
        overwrite : bool, default: False
            If True, remove existing value for the simulationId, year and
            measurementId before insert

        Raises
        ------
        AssertionError
            If an entry exists for the simulationId in the database table
            Convergence using overwriting=False
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(convergence) is bool
        assert type(overwrite) is bool

        sqlcommand = 'SELECT simulationId FROM Convergence WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Convergence WHERE simulationId = ?'
                self._c.execute(sqlcommand, (simulationId, ))
        else:
            assert len(dataset) == 0

        #Insert for the tracer norm
        purchases = []
        purchases.append((simulationId, int(convergence)))
        self._c.executemany('INSERT INTO Convergence VALUES (?,?)', purchases)
        self._conn.commit()


    def _create_table_tracerNorm(self, norm='2', trajectory=''):
        """
        Create table tracerNorm

        Parameters
        ----------
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        """
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        if trajectory == '':
            self._c.execute('''CREATE TABLE Tracer{:s}{:s}Norm (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, tracer REAL NOT NULL, N REAL NOT NULL, DOP REAL, P REAL, Z REAL, D REAL, PRIMARY KEY (simulationId, year))'''.format(trajectory, norm))
        else:
            self._c.execute('''CREATE TABLE Tracer{:s}{:s}Norm (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, timestep INTEGER NOT NULL, tracer REAL NOT NULL, N REAL NOT NULL, DOP REAL, P REAL, Z REAL, D REAL, PRIMARY KEY (simulationId, year, timestep))'''.format(trajectory, norm))


    def check_tracer_norm(self, simulationId, expectedCount, timestep=1, norm='2', trajectory=''):
        """
        Check the number of tracer norm entries for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up
        timestep : int, default: 1
            Timestep used for the spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT COUNT(*) FROM Tracer{:s}{}Norm WHERE simulationId = ?{:s}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
        self._c.execute(sqlcommand, (simulationId, ) if trajectory == '' else (simulationId, timestep))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerNorm: Expected: {} Get: {} (Norm: {})***'.format(expectedCount, count[0][0], norm))

        return count[0][0] == expectedCount


    def insert_tracer_norm_tuple(self, simulationId, year, tracer, N, DOP=None, P=None, Z=None, D=None, timestep=1, norm='2', trajectory='', overwrite=False):
        """
        Insert tracer norm value

        Insert tracer norm value. If a database entry of the tracer norm for
        the simulationId and year already exists, the existing entry is
        deleted and the new one is inserted (if the flag overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation
        tracer : float
            Norm including all tracers
        N : float
            Norm of the N tracer
        DOP : None or float, default: None
            Norm of the DOP tracer. None, if the biogeochemical model does not
            contain the DOP tracer
        P : None or float, default: None
            Norm of the P tracer. None, if the biogeochemical model does not
            contain the P tracer
        Z : None or float, default: None
            Norm of the Z tracer. None, if the biogeochemical model does not
            contain the Z tracer
        D : None or float, default: None
            Norm of the D tracer. None, if the biogeochemical model does not
            contain the D tracer
        timestep : int, default: 1
            Timestep used for the spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(tracer) is float
        assert type(N) is float
        assert DOP is None or type(DOP) is float
        assert P is None or type(P) is float
        assert Z is None or type(Z) is float
        assert D is None or type(D) is float
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?{}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
        self._c.execute(sqlcommand_select, (simulationId, year) if trajectory == '' else (simulationId, year, timestep))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?{}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
                self._c.execute(sqlcommand, (simulationId, year) if trajectory == '' else (simulationId, year, timestep))
        else:
            assert len(dataset) == 0

        #Generate insert for the tracer norm
        purchases = []
        purchases.append((simulationId, year, tracer, N, DOP, P, Z, D) if trajectory == '' else (simulationId, year, timestep, tracer, N, DOP, P, Z, D))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO Tracer{}{}Norm VALUES (?,?,?,?,?,?,?,?{})'.format(trajectory, norm, '' if trajectory == '' else ',?'), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_tracer_norm(self, simulationId, timestep=1, norm='2', trajectory=''):
        """
        Delete entries of the tracer norm

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        timestep : int, default: 1
            Timestep used for the spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        """
        assert type(simulationId) is int and simulationId >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'DELETE FROM Tracer{}{}Norm WHERE simulationId = ?{:s}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
        self._c.execute(sqlcommand, (simulationId,) if trajectory == '' else (simulationId,timestep))
        self._conn.commit()


    def _create_table_tracerDifferenceNorm(self, norm='2', trajectory=''):
        """
        Create table TracerDifferenceNorm

        Parameters
        ----------
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        """
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        if trajectory == '':
            self._c.execute('''CREATE TABLE TracerDifference{:s}{:s}Norm (simulationIdA INTEGER NOT NULL REFERENCES Simulation(simulationId), simulationIdB INTEGER NOT NULL REFERENCES Simulation(simulationId), yearA INTEGER NOT NULL, yearB INTEGER NOT NULL, tracer REAL NOT NULL, N REAL NOT NULL, DOP REAL, P REAL, Z REAL, D REAL, PRIMARY KEY (simulationIdA, simulationIdB, yearA, yearB))'''.format(trajectory, norm))
        else:
            self._c.execute('''CREATE TABLE TracerDifference{:s}{:s}Norm (simulationIdA INTEGER NOT NULL REFERENCES Simulation(simulationId), simulationIdB INTEGER NOT NULL REFERENCES Simulation(simulationId), yearA INTEGER NOT NULL, yearB INTEGER NOT NULL, timestep INTEGER NOT NULL, tracer REAL NOT NULL, N REAL NOT NULL, DOP REAL, P REAL, Z REAL, D REAL, PRIMARY KEY (simulationIdA, simulationIdB, yearA, yearB, timestep))'''.format(trajectory, norm))


    def check_difference_tracer_norm(self, simulationIdA, simulationIdB, expectedCount, timestep=1, norm='2', trajectory=''):
        """
        Check number of tracer difference norm entries for the simulationId

        Parameters
        ----------
        simulationIdA : int
            Id defining the parameter for spin up calculation
        simulationIdB : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up
        timestep : int, default: 1
            Timestep used for the spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationIdA) is int and simulationIdA >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert type(expectedCount) is int and expectedCount >= 0
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT COUNT(*) FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ?{:s}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
        self._c.execute(sqlcommand, (simulationIdA, simulationIdB) if trajectory == '' else (simulationIdA, simulationIdB, timestep))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerDifferenceNorm: Expected: {} Get: {} (Norm: {})***'.format(expectedCount, count[0][0], norm))

        return count[0][0] == expectedCount


    def insert_difference_tracer_norm_tuple(self, simulationIdA, simulationIdB, yearA, yearB, tracerDifferenceNorm, N, DOP=None, P=None, Z=None, D=None, timestep=1, norm='2', trajectory='', overwrite=False):
        """
        Insert the norm of a difference between two tracers

        Insert the norm of a difference between two tracers. If a database
        entry of the norm between the tracers of the simulations with the
        simulationIdA and simulationIdB as well as yearA and yearB already
        exists, the existing entry is deleted and the new one is inserted (if
        the flag overwrite is True).

        Parameters
        ----------
        simulationIdA : int
            Id defining the parameter for spin up calculation of the first
            used tracer
        simulationIdB : int
            Id defining the parameter for spin up calculation of the second
            used tracer
        yearA : int
            Model year of the spin up calculation for the first tracer
        yearB : int
            Model year of the spin up calculation for the second tracer
        tracerDifferenceNorm : float
            Norm including all tracers
        N : float
            Norm of the N tracer
        DOP : None or float, default: None
            Norm of the DOP tracer. None, if the biogeochemical model does not
            contain the DOP tracer
        P : None or float, default: None
            Norm of the P tracer. None, if the biogeochemical model does not
            contain the P tracer
        Z : None or float, default: None
            Norm of the Z tracer. None, if the biogeochemical model does not
            contain the Z tracer
        D : None or float, default: None
            Norm of the D tracer. None, if the biogeochemical model does not
            contain the D tracer
        timestep : int, default: 1
            Timestep used for the spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        overwrite : bool, default: False
            If True, overwrite an existing entry.

        Raises
        ------
        AssertionError
            If an existing entry exists and should not be overwritten.
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts.

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try.
        """
        assert type(simulationIdA) is int and simulationIdA >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert type(yearA) is int and yearA >= 0
        assert type(yearB) is int and yearB >= 0
        assert type(tracerDifferenceNorm) is float
        assert type(N) is float
        assert DOP is None or type(DOP) is float
        assert P is None or type(P) is float
        assert Z is None or type(Z) is float
        assert D is None or type(D) is float
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationIdA, simulationIdB, yearA, yearB FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?{:s}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
        if overwrite:
            #Test, if dataset for this simulationIdA, simulationIdB, yearA and yearB combination exists
            self._c.execute(sqlcommand_select, (simulationIdA, simulationIdB, yearA, yearB) if trajectory == '' else (simulationIdA, simulationIdB, yearA, yearB, timestep))
            dataset = self._c.fetchall()
            #Remove database entry for this simulationIdA, simulationIdB, yearA and yearB
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?{:s}'.format(trajectory, norm, '' if trajectory == '' else ' AND timestep = ?')
                self._c.execute(sqlcommand, (simulationIdA, simulationIdB, yearA, yearB) if trajectory == '' else (simulationIdA, simulationIdB, yearA, yearB, timestep))
        else:
            self._c.execute(sqlcommand_select, (simulationIdA, simulationIdB, yearA, yearB) if trajectory == '' else (simulationIdA, simulationIdB, yearA, yearB, timestep))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate insert for the tracer norm
        purchases = []
        purchases.append((simulationIdA, simulationIdB, yearA, yearB, tracerDifferenceNorm, N, DOP, P, Z, D) if trajectory == '' else (simulationIdA, simulationIdB, yearA, yearB, timestep, tracerDifferenceNorm, N, DOP, P, Z, D))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO TracerDifference{}{}Norm VALUES (?,?,?,?,?,?,?,?,?,?{:s})'.format(trajectory, norm, '' if trajectory == '' else ',?'), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def get_parameterId_for_simid(self, simulationId):
        """
        Returns parameterId for the simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        int
            parameterId for the simulationId

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT parameterId FROM Simulation WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        parameterId = self._c.fetchall()
        assert len(parameterId) == 1
        return int(parameterId[0][0])


    def get_simids_for_parameter(self, parameterId):
        """
        Returns all simulationIds for the given parameterId

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example

        Returns
        -------
        list [int]
            List of all simulationId whose spin up calculations use the model
            parameters with the given parameterId
        """
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)

        sqlcommand = 'SELECT simulationId FROM simulation WHERE parameterId = ? ORDER BY simulationId;'
        self._c.execute(sqlcommand, (parameterId,))
        simidList = []
        for parameterId in self._c.fetchall():
            simidList.append(parameterId[0])
        return simidList


    def get_simids_timestep_for_parameter(self, parameterId):
        """
        Returns simulationIds and time steps for the given parameterId

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example

        Returns
        -------
        list [list [int]]
            List of all combination of simulationId and time step whose spin
            up calculations use the model parameters with the given
            parameterId
        """
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)

        sqlcommand = 'SELECT simulationId, timestep FROM simulation WHERE parameterId = ? ORDER BY simulationId;'
        self._c.execute(sqlcommand, (parameterId,))
        simidList = []
        for parameterId in self._c.fetchall():
            simidList.append([parameterId[0], parameterId[1]])
        return simidList


    def get_simids_timestep_for_parameter_model(self, parameterId, model, concentrationId=None):
        """
        Returns simulationIds and time steps for the model and parameterId

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        model : str
            Name of the biogeochemical model
        concentrationId : int or None, default: None
            Id of the initial concentration. If None, use default constant
            initial concentration of the model

        Returns
        -------
        list [list [int]]
            List of all combination of simulationId and time step whose spin
            up calculations use the model parameters with the given
            parameterId and model
        """
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId

        sqlcommand = 'SELECT simulationId, timestep FROM Simulation WHERE parameterId = ? AND model = ? AND concentrationId = ? ORDER BY simulationId;'
        self._c.execute(sqlcommand, (parameterId, model, concentrationId if concentrationId is not None else Timesteps_Constants.CONCENTRATIONID_DICT[model]))
        simidList = []
        for parameterId in self._c.fetchall():
            simidList.append([parameterId[0], parameterId[1]])
        return simidList


    def get_simids_for_model_timestep(self, model, timestep):
        """
        Returns simulationIds for the model and timestep

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : int
            Timestep used for the spin up calculation

        Returns
        -------
        list [int]
            List all the simulationIds
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND timestep = ? ORDER BY simulationId;'
        self._c.execute(sqlcommand, (model, timestep))
        simidList = []
        for simulationId in self._c.fetchall():
            simidList.append(simulationId[0])
        return simidList


    def read_costfunction_values_count(self, simulationId):
        """
        Returns the count of the cost function values for a simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        int
            Count of cost function values for the given simulationId

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT COUNT(*) FROM CostfuctionEvaluation WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId,))
        count = self._c.fetchall()
        assert len(count) == 1
        return count[0][0]


    def read_costfunction_values(self, simulationId):
        """
        Returns the cost function values for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        list [list [int]]
            List of a lists including year, ols, wls and gls for the given
            simulationId
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT year, OLS, WLS, GLS FROM CostfuctionEvaluation WHERE simulationId = ? ORDER BY year;'
        self._c.execute(sqlcommand, (simulationId,))
        simrows = self._c.fetchall()
        simlen = len(simrows)
        simdata = np.empty(shape=(simlen, 4))
        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1], row[2], row[3]])
            i = i+1
        return simdata


    def read_costfunction_values_for_simid_list(self, simulationIdList, count=200):
        """
        Returns cost function values for the given list of simulationIds


        Parameters
        ----------
        simulationIdList : list [int]
            List of Ids defining the parameter for spin up calculation
        count : int, default: 200
            Expected number of database entries in the table
            CostfunctionEvaluation for each simulationId in the given
            simulationIdList

        Returns
        -------
        numpy.ndarray
            Numpy array with year and cost function values ols, wls and gls
            for each simulationId in the simulationIdList

        Notes
        -----
        This function writes a warning to the log if the expected number of
        entries (see parameter count) is not present.
        """
        assert type(simulationIdList) is list
        assert type(count) is int and 0 < count

        simdata = np.zeros(shape=(len(simulationIdList), count, 4))
        i = 0
        for simid in simulationIdList:
            if self.read_costfunction_values_count(simid) == count:
                simdata[i,:,:] = self.read_costfunction_values(simid)
                i = i+1
            else:
                logging.warning('Number of datasets in the table CostfunctionEvaluation for simId {:d} is {:d} and not {:d}'.format(simid, self.read_costfunction_values_count(simid), count))
        return simdata


    def get_timestep_for_simid(self, simulationId):
        """
        Returns the time step for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        int
            Used time step of the spin up calculation with the given
            simulationId

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT timestep FROM Simulation WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId,))
        timestep = self._c.fetchall()
        assert len(timestep) == 1
        return timestep[0][0]


    def read_spinup_values_for_simid(self, simulationId):
        """
        Returns values of the spin up for a given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        numpy.ndarray
            2D array containing the year and tolerance, respectively
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT year, tolerance FROM Spinup WHERE simulationId = ? ORDER BY year;'
        self._c.execute(sqlcommand,  (simulationId, ))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_spinup_year(self, model, concentrationId):
        """
        Read the required year to reach the given tolerance
        """
        #TODO
        pass


    def read_spinup_tolerance(self, model, concentrationId, year):
        """
        Read the spin up tolerance for the given model year
        """
        #TODO
        pass


    def read_rel_norm(self, model, concentrationId, year=None, norm='2', parameterId=None, trajectory=''):
        """
        Read for every parameterId the norm value for the given annId from the database.
        If parameterId is not None, read only the relative difference for the given parameterId.
        @author: Markus Pfeil
        """
        #TODO
        pass


    def read_spinup_years_for_timestep_and_tolerance(self, timestep, tolerance, model, lhs=True):
        """
        Get the year to reach a given spinup tolerance tol for all parameter sets simulated with the given model and time step
        @author: Markus Pfeil
        """
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tolerance) is float and tolerance >= 0
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND simulation.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT spinup.simulationId as simulationId, MIN(spinup.year) AS year, spinup.tolerance AS tolerance FROM Spinup, Simulation, Convergence WHERE simulation.timestep = ? AND simulation.model = ?{:s} AND simulation.simulationId = spinup.simulationId AND simulation.simulationId = convergence.simulationId AND convergence.convergence = 1 AND spinup.tolerance <= ? GROUP BY spinup.simulationId;'.format(parameterStr)
        self._c.execute(sqlcommand,  (timestep, model, tolerance))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 3))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1], row[2]])
            i = i+1
        return simdata


    def read_spinup_avg_year_for_tolerance(self, tolerance, model, lhs=True):
        """
        Get the average of the years to reach the given spinup tolerance tol over all parameter sets for every time step simulated with the given model.
        @author: Markus Pfeil
        """
        assert type(tolerance) is float and tolerance >= 0
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.timestep as timestep, AVG(sp.year) as year FROM Simulation AS sim, Spinup AS sp, Convergence AS con WHERE sim.model = ?{:s} AND sim.simulationId = sp.simulationId AND sim.simulationId = con.simulationId AND con.convergence = 1 AND sp.tolerance <= ? AND NOT EXISTS ( SELECT * FROM spinup AS sp1 WHERE sp1.simulationId = sim.simulationId AND sp1.tolerance <= ? AND sp1.year < sp.year) GROUP BY sim.timestep;'.format(parameterStr)
        self._c.execute(sqlcommand,  (model, tolerance, tolerance))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_spinup_years_for_tolerance_timestep(self, tolerance, model, timestep, lhs=True):
        """
        Get an array of the years to reach the given spinup tolerance tol over every parameterId of the simulationd using the given model and time step, except of simulations without convergence
        @author: Markus Pfeil
        """
        assert type(tolerance) is float and tolerance >= 0
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId as simid, sp.year as year FROM Simulation AS sim, Spinup AS sp, Convergence as con WHERE sim.model = ?{:s} AND sim.simulationId = con.simulationId AND con.convergence = 1 AND sim.simulationId = sp.simulationId AND sp.tolerance <= ? AND sim.timestep = ? AND NOT EXISTS ( SELECT * FROM spinup AS sp1 WHERE sp1.simulationId = sp.simulationId AND sp1.tolerance <= ? AND sp1.year < sp.year)'.format(parameterStr)
        self._c.execute(sqlcommand,  (model, tolerance, timestep, tolerance))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows)))

        i = 0
        for row in simrows:
            simdata[i] = row[1]
            i = i+1
        return simdata


    def read_spinup_tolerance_for_year_timestep(self, model, timestep, year=9999, lhs=True):
        """
        Get an array with the reached tolerances of the spin up after the given year of all parameterIds simulated with the given model and time step, except of the parameterId 0 (reference parameter set).
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId AS simid, sp.tolerance AS tolerance FROM Simulation AS sim, Spinup AS sp, Convergence AS con WHERE sim.model = ?{:s} AND sim.timestep = ? AND sim.simulationId = con.simulationId AND con.convergence = 1 AND sim.simulationId = sp.simulationId AND sp.year = ? ORDER BY sim.simulationId;'
        self._c.execute(sqlcommand,  (model, timestep, year)).format(parameterStr)
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows)))

        i = 0
        for row in simrows:
            simdata[i] = row[1]
            i = i+1
        return simdata


    def read_spinup_count_tolerance(self, tolerance, model):
        """
        Get an array with the reached spinup tolerance for every time step over all parameter sets simulated with the given model.
        @author: Markus Pfeil
        """
        assert type(tolerance) is float and tolerance >= 0
        assert model in Metos3d_Constants.METOS3D_MODELS

        sqlcommand = 'SELECT sim.timestep AS model, count(*) FROM Spinup as sp, Simulation as sim WHERE sim.model = ? AND sim.simulationId = sp.simulationId AND sp.tolerance <= ? AND NOT EXISTS ( SELECT * FROM spinup AS sp1 WHERE sp1.simulationId = sp.simulationId AND sp1.tolerance <= ? AND sp1.year < sp.year) GROUP BY sim.timestep;'
        self._c.execute(sqlcommand,  (model, tolerance, tolerance))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i][0] = row[0]
            simdata[i][1] = row[1]
            i = i+1
        return simdata


    def read_spinupTolerance(self, simulationId, year=9999):
        """
        Returns the spin up tolerance of a simulation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int, default: 9999
            Model year of the simulatin

        Returns
        -------
        float
            Spin up tolerance of the simulation for the given model year
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and 0 <= year

        sqlcommand = 'SELECT tolerance FROM Spinup WHERE simulationId = ? AND year = ?;'
        self._c.execute(sqlcommand,  (simulationId, year))
        tolerance = self._c.fetchall()
        assert len(tolerance) == 1
        return float(tolerance[0][0])


    def oscillation_spin(self, simulationId, startYear=5000, deviation=0.05, percentageDeviation=0.1, count=0.1):
        """
        Returns a boolean whether the spin up norm oscillates

        Returns a boolean whether the spin up norm oscillates. For checking whether the spin up calculation oscillates, the average spin up norm is calculated using the model years starting from the given startYear.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        startYear : int, default 5000
            Check the spin up norm for oscillation between the startYear and
            the last year of the spin up calculation
        deviation : float, default: 0.1
            Percentage deviation from the norm value of the spin up norm that
            must be undercut or exceeded
        percentageDeviation : float, default: 0.1
            Proportion of the model years for which the percentage deviation
            from the mean value must be undercut or exceeded
        count : float, default: 0.1
            Percentage of the model years for which the spin up norm changes
            from a value less than the mean to greater than the mean or vice
            versa

        Returns
        -------
        bool
            True if the spin up oscillates otherweise False
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(startYear) is int and 0 <= startYear
        assert type(deviation) is float and 0.0 <= deviation and deviation <= 1.0
        assert type(count) is float and 0.0 <= count and count <= 1.0

        sqlcommand = 'SELECT tolerance FROM spinup WHERE simulationId = ? AND year >= ? ORDER BY year;'
        self._c.execute(sqlcommand,  (simulationId, startYear))
        simrows = self._c.fetchall()
        data = np.empty(shape=len(simrows))
        i = 0
        for row in simrows:
            data[i] = row[0]
            i = i + 1

        if len(data) > 0:
            mean = np.mean(data)
            countGreaterMean = (data > (mean + deviation * mean)).sum()
            countLessMean = (data < (mean - deviation * mean)).sum()

            countChange = 0
            spinupNormValue = data[0]
            for i in range(1, len(data)):
                if spinupNormValue <= mean and mean < data[i]:
                    countChange = countChange + 1
                elif data[i] <= mean and mean < spinupNormValue:
                    countChange = countChange + 1
                spinupNormValue = data[i]

            oscillation = (int(len(data) * percentageDeviation) < countGreaterMean) and (int(len(data) * percentageDeviation) < countLessMean) and (int(len(data) * count) < countChange)
        else:
            oscillation = False

        return oscillation


    def read_tracer_norm_values_for_simid(self, simulationId, norm='2', trajectory=''):
        """
        Returns norm values for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        numpy.ndarray
            2D array with the year and the norm value
        """
        assert type(simulationId) is int and simulationId >= 0
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT year, tracer FROM Tracer{}{}Norm WHERE simulationId = ? ORDER BY year;'.format(trajectory, norm)
        self._c.execute(sqlcommand,  (simulationId, ))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_tracer_norm_value_for_simid_year(self, simulationId, year, norm='2', trajectory=''):
        """
        Return norm value for the given simulationId and year

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the calculated tracer concentration
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        float
            Norm value

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT tracer FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?'.format(trajectory, norm)
        self._c.execute(sqlcommand,  (simulationId, year))
        simdata = self._c.fetchall()
        assert len(simdata) == 1
        return simdata[0][0]


    def read_tracer_difference_norm_values_for_simid(self, simulationId, simulationIdB, yearB=None, norm='2', trajectory=''):
        """
        Returns norm values of the difference for two simulations

        Returns the norm values for the difference of the tracer concentration
        between the spin up calculations with simulationId and simulationIdB
        as reference solution.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        simulationIdB : int
            Id defining the parameter for spin up calculation for the reference
            solution
        yearB : int or None, default: None
            Model year of the calculated tracer concentration for the reference
            solution. If None, use the same year for both spin up calculations
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        numpy.ndarray
            2D array with the year and the norm value
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert yearB is None or type(yearB) is int and 0 <= yearB
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        if yearB is None:
            sqlcommand = 'SELECT yearA, tracer FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = yearB ORDER BY yearA;'.format(trajectory, norm)
            self._c.execute(sqlcommand,  (simulationId, simulationIdB))
        else:
            sqlcommand = 'SELECT yearA, tracer FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearB = ? ORDER BY yearA;'.format(trajectory, norm)
            self._c.execute(sqlcommand,  (simulationId, simulationIdB, yearB))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_tracer_difference_norm_value_for_simid_year(self, simulationId, simulationIdB, year, yearB=None, norm='2', trajectory=''):
        """
        Returns norm values of the difference for one model year

        Returns the norm values for the difference of the tracer concentration
        between the spin up calculations with simulationId and simulationIdB
        as reference solution for one model year.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        simulationIdB : int
            Id defining the parameter for spin up calculation for the reference
            solution
        year : int
            Model year of the calculated tracer concentration
        yearB : int or None, default: None
            Model year of the calculated tracer concentration for the reference
            solution. If None, use the same year for both spin up calculations
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory

        Returns
        -------
        float
            Norm value

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert type(year) is int and 0 <= year
        assert yearB is None or type(yearB) is int and 0 <= yearB
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        if yearB is None:
            sqlcommand = 'SELECT tracer FROM TracerDifference{}{}Norm WHERE simulationId = ? AND simulationIdB = ? AND yearA = ? AND yearB = yearA'.format(trajectory, norm)
            self._c.execute(sqlcommand,  (simulationId, simulationIdB, year))
        else:
            sqlcommand = 'SELECT tracer FROM TracerDifference{}{}Norm WHERE simulationId = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?'.format(trajectory, norm)
            self._c.execute(sqlcommand,  (simulationId, simulationIdB, year, yearB))
        simdata = self._c.fetchall()
        assert len(simdata) == 1
        return simdata[0][0]


    def read_tracer_norm_rel_error_values_model_timestep(self, model, timestep, year=10000, norm='2', trajectory='', lhs=True):
        """
        Read norm values for the given model, time step and year in the given norm from the database.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT diff_norm.simulationId as simulationId, diff_norm.tracer/norm.tracer as Error FROM Tracer{}{}Norm AS diff_norm, Simulation AS sim, Simulation AS sim2, Tracer{}{}Norm AS norm, Convergence AS con, Convergence AS con2 WHERE sim.model = ? AND sim.timestep = ?{:s} AND diff_norm.year = ? AND sim.simulationId = diff_norm.simulationId AND sim.simulationId = con.simulationId AND con.convergence = 1 AND sim2.timestep = 1 AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.simulationId = con2.simulationId AND con2.convergence = 1 AND sim2.simulationId = norm.simulationId AND norm.year = ? ORDER BY norm.simulationId;'.format(trajectory, norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand,  (model, timestep, year, year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1

        return simdata


    def read_tracer_difference_norm_rel_error_values_model_timestep(self, model, timestep, year=10000, yearB=10000, norm='2', trajectory='', lhs=True):
        """
        Read norm values for the given model, time step and year in the given norm from the database.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert type(yearB) is int and 0 <= yearB
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT diff_norm.simulationIdA as simulationId, diff_norm.tracer/norm.tracer as Error FROM TracerDifference{}{}Norm AS diff_norm, Simulation AS sim, Simulation AS sim2, Tracer{}{}Norm AS norm, Convergence AS con, Convergence AS con2 WHERE sim.model = ? AND sim.timestep = ?{:s} AND diff_norm.yearA = ? AND diff_norm.yearB = ? AND sim.simulationId = diff_norm.simulationIdA AND sim.simulationId = con.simulationId AND con.convergence = ? AND sim2.simulationId = diff_norm.simulationIdB AND sim2.timestep = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.simulationId = con2.simulationId AND con2.convergence = ? AND sim2.simulationId = norm.simulationId AND norm.year = ? ORDER BY norm.simulationId;'.format(trajectory, norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand,  (model, timestep, year, yearB, int(True), 1, int(True),  year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1

        return simdata


    def read_norm_avg_error_model(self, model, year=10000, norm='2', trajectory='', lhs=True):
        """
        Return the average of the relative error over all parameterIds of the tracer norm for every time step for the given model
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.timestep as timestep, AVG(diff_norm.tracer/norm.tracer) as Error FROM Tracer{}{}Norm AS diff_norm, Simulation AS sim, Simulation AS sim2, Tracer{}{}Norm AS norm, Convergence AS con, Convergence AS con2 WHERE sim.model = ?{:s} AND diff_norm.year = ? AND sim.simulationId = con.simulationId AND con.convergence = ? AND sim.simulationId = diff_norm.simulationId AND sim2.timestep = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND con2.simulationId = sim2.simulationId AND con2.convergence = ? AND sim2.simulationId = norm.simulationId AND norm.year = ? GROUP BY sim.timestep;'.format(trajectory, norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand,  (model, year, int(True), 1, int(True), year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1

        return simdata



    def read_difference_norm_avg_error_model(self, model, year=10000, yearB=10000, norm='2', trajectory='', lhs=True):
        """
        Return the average of the relative error over all parameterIds of the tracer norm for every time step for the given model
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert type(yearB) is int and 0 <= yearB
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.timestep as timestep, AVG(diff_norm.tracer/norm.tracer) as Error FROM TracerDifference{}{}Norm AS diff_norm, Simulation AS Sim, Simulation AS sim2, Tracer{}{}Norm AS norm, Convergence AS con, Convergence AS con2 WHERE sim.model = ?{:s} AND diff_norm.year = ? AND diff_norm.yearB = ? AND sim.simulationId = con.simulationId AND con.convergence = ? AND sim.simulationId = diff_norm.simulationIdA AND sim2.simulationId = diff_norm.simulationIdB AND sim2.timestep = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND con2.simulationId = sim2.simulationId AND con2.convergence = ? AND sim2.simulationId = norm.simulationId AND norm.year = ? GROUP BY sim.timestep;'.format(trajectory, norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand,  (model, year, yearB, int(True), 1, int(True), year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1

        return simdata


    def read_rel_norm_for_model_timestep(self, model, timestep, year=10000, norm='2', trajectory='', lhs=True):
        """
        Read for every parameterId the norm value for the given model, time step and year from the database.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId as simid, diff_norm.tracer/norm.tracer as Error FROM Tracer{}{}Norm AS diff_norm, Simulation AS sim, Simulation AS sim2, Tracer{}{}Norm AS norm, Convergence AS con WHERE sim.model = ? AND sim.timestep = ?{} AND sim.simulationId = con.simulationId and con.convergence = ? AND sim.simulationId = diff_norm.simulationId AND diff_norm.year = ? AND sim2.timestep = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.simulationId = norm.simulationId AND norm.year = ?'.format(trajectory, norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand, (model, timestep, int(True), year, 1, year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows)))

        i = 0
        for row in simrows:
            simdata[i] = row[1]
            i = i+1

        return simdata


    def read_difference_rel_norm_for_model_timestep(self, model, timestep, year=10000, yearB=10000, norm='2', trajectory='', lhs=True):
        """
        Read for every parameterId the norm value for the given model, time step and year from the database.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert type(yearB) is int and 0 <= yearB
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId AS simid, diff_norm.tracer/norm.tracer AS Error FROM TracerDifference{}{}Norm AS diff_norm, Simulation AS sim, Simulation AS sim2, Tracer{}{}Norm AS norm, Convergence AS con WHERE sim.model = ? AND sim.timestep = ?{} AND sim.simulationId = con.simulationId and con.convergence = ? AND sim.simulationId = diff_norm.simulationIdA AND diff_norm.year = ? AND diff_norm.yearB = ? AND diff_norm.simulationId = sim2.simulationId AND sim2.timestep = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.simulationId = norm.simulationId AND norm.year = ?'.format(trajectory, norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand, (model, timestep, int(True), year, yearB, 1, year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows)))

        i = 0
        for row in simrows:
            simdata[i] = row[1]
            i = i+1

        return simdata


    def read_spinupNom_relNorm_for_model_year(self, model, timestep, year=10000, norm='2', trajectory='', lhs=True):
        """
        Returns spin up norm and norm values

        Returns the spin up norm values and the relative error (the norm of
        the tracer concentration difference between the spin up calculation
        using the given time step and the spin up calculation using the time
        step 1dt) for every parameter and the given model. 

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : int
            Time step used for the spin up calculation
        year : int, default: 10000
            Model year of the calculated tracer concentration for the spin up
            calculation. For the spin up norm the previous model year is used.
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        lhs : bool, default: True
            Use only the model parameter of the latin hypercube sample

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId, parameterId, the spin up norm and
            the relative error
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId, sim.parameterId, sp.tolerance, normDiff.tracer/norm.tracer FROM Simulation AS sim, Simulation AS sim2, Spinup AS sp, Tracer{:s}Norm AS norm, TracerDifference{:s}{:s}Norm AS normDiff WHERE sim.model = ? AND sim.timestep = ?{:s} AND sim.simulationId = sp.simulationId AND sp.year = ? AND sim2.simulationId = norm.simulationId AND norm.year = ? AND normDiff.simulationIdA = sim.simulationId AND normDiff.yearA = ? AND normDiff.yearB = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.timestep = 1 AND normDiff.simulationIdB = sim2.simulationId ORDER BY sim.parameterId;'.format(norm, trajectory, norm, parameterStr)
        self._c.execute(sqlcommand, (model, timestep, year-1, year, year, year))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 4))

        i = 0
        for row in simrows:
            simdata[i, 0] = row[0]
            simdata[i, 1] = row[1]
            simdata[i, 2] = row[2]
            simdata[i, 3] = row[3]
            i = i+1

        return simdata


    def read_tolerance_required_ModelYears_relNorm(self, model, timestep, tolerance=0.0001, norm='2', trajectory='', lhs=True):
        """
        Returns required model years and norm values

        Returns the required model years to reach the given tolerance during
        the spin up and the relative error (the norm of the tracer
        concentration difference between the spin up calcluation using the
        given time step (stopped with the given tolerance) and the spin up using
        the time step 1dt over 10000 model years) for every parameter and the
        given model.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : int
            Time step used for the spin up calculation
        tolerance : float, default: 0.0001
            Tolerance used for the spin up calculation
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        lhs : bool, default: True
            Use only the model parameter of the latin hypercube sample

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId, parameterId, the required model year
            and the relative error
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tolerance) is float and 0.0 <= tolerance
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId, sim.parameterId, sp.year, normDiff.tracer/norm.tracer FROM Simulation AS sim, Simulation AS sim2, Spinup AS sp, Tracer{:s}Norm AS norm, TracerDifference{:s}{:s}Norm AS normDiff WHERE sim.model = ? AND sim.timestep = ?{:s} AND sim.simulationId = sp.simulationId AND sp.tolerance < ? AND NOT EXISTS (SELECT * FROM Spinup AS sp2 WHERE sp.simulationId = sp2.simulationId AND sp2.tolerance < ? AND sp.year > sp2.year) AND sim2.simulationId = norm.simulationId AND norm.year = ? AND normDiff.simulationIdA = sim.simulationId AND normDiff.simulationIdB = sim2.simulationId AND normDiff.yearB = ? AND sim.model = sim2.model AND sim.parameterId = sim2.parameterId AND sim.concentrationId = sim2.concentrationId AND sim2.timestep = ? AND normDiff.yearA >= sp.year AND NOT EXISTS (SELECT * FROM TracerDifference{:s}{:s}Norm AS normDiff2 WHERE normDiff.simulationIdA = normDiff2.simulationIdA AND normDiff.simulationIdB = normDiff2.simulationIdB AND normDiff.yearB = normDiff2.yearB AND normDiff2.yearA >= sp.year AND normDiff2.yearA < normDiff.yearA) ORDER BY sim.parameterId;'.format(norm, trajectory, norm, parameterStr, trajectory, norm)
        self._c.execute(sqlcommand, (model, timestep, tolerance, tolerance, 10000, 10000, 1))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 4))

        i = 0
        for row in simrows:
            simdata[i, 0] = row[0]
            simdata[i, 1] = row[1]
            simdata[i, 2] = row[2]
            simdata[i, 3] = row[3]
            i = i+1

        return simdata


    def read_costfunction_relNorm(self, model, timestep, year=10000, costfunction='OLS', measurementId=0, lhs=True):
        """
        Returns cost function and spin-up tolerance values

        Returns the cost function value and spin-up tolerance value for every
        parameter and the given model.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : int
            Time step used for the spin up calculation
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        costfunction : {'OLS', 'GLS', 'WLS'}, default: 'OLS'
            Type of the cost function
        measurementId : int, default: 0
            Selection of the tracer included in the cost function calculation
        lhs : bool, default: True
            Use only the model parameter of the latin hypercube sample

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId, parameterId, cost function and
            tolerance of the spin-up
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'GLS', 'WLS']
        assert type(measurementId) is int and 0 <= measurementId
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId, sim.parameterId, c.{:s}, sp.tolerance FROM Simulation AS sim, CostfuctionEvaluation AS c, Spinup AS sp WHERE sim.model = ? AND sim.timestep = ?{:s} AND sim.simulationId = c.simulationId and c.year = ? AND c.measurementId = ? AND sim.simulationId = sp.simulationId AND sp.year = ? ORDER BY sim.parameterId;'.format(costfunction, parameterStr)
        self._c.execute(sqlcommand, (model, timestep, year, measurementId, year-1))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 4))

        i = 0
        for row in simrows:
            simdata[i, 0] = row[0]
            simdata[i, 1] = row[1]
            simdata[i, 2] = row[2]
            simdata[i, 3] = row[3]
            i = i+1

        return simdata


    def read_costfunction_values_for_simid(self, simulationId, costfunction='OLS', measurementId=0):
        """
        Read all costfunction values for the given simulationId
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0

        sqlcommand = 'SELECT year, {} FROM CostfuctionEvaluation WHERE simulationId = ? AND measurementId = ? ORDER BY year;'.format(costfunction)
        self._c.execute(sqlcommand,  (simulatonId, measurementId))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1

        return simdata


    def read_costfunction_difference_values_for_simid(self, simulationId, costfunction='OLS', measurementId=0, year=None):
        """
        Read the difference between the costfunction values for the given simulationId and the simulation calculated with time step 1
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0
        assert year is None or type(year) is int and 0 <= year

        sqlcommand = 'SELECT c.year AS year, c.{}-d.{} AS Difference FROM CostfuctionEvaluation AS c, CostfuctionEvaluation AS d, Simulation AS sim_c, Simulation AS sim_d WHERE c.simulationId = ? AND c.measurementId = ? AND sim_c.simulationId = c.simulationId AND sim_d.timestep = ? AND sim_c.model = sim_d.model AND sim_c.parameterId = sim_d.parameterId AND sim_c.concentrationId = sim_d.concentrationId AND d.simulationId = sim_d.simulationId AND d.measurementId = ? AND d.year = {:s} ORDER BY c.year;'.format(costfunction, costfunction, 'c.year' if year is not None else '{:d}'.format(year))
        self._c.execute(sqlcommand,  (simulationId, measurementId, 1, measurementId))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_costfunction_values_for_simid_year(self, simulationId, year, costfunction='OLS', measurementId=0):
        """
        Read the costfunction value for the given simulationId and year from the database.
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0

        sqlcommand = 'SELECT {} FROM CostfuctionEvaluation WHERE simulationId = ? AND year = ? AND measurementId = ?;'.format(costfunction)
        self._c.execute(sqlcommand,  (simulationId, year, measurementId))
        simdata = self._c.fetchall()
        assert len(simdata) == 1
        return simdata[0][0]


    def read_costfunction_rel_error_values_model_timestep(self, model, timestep, year=10000, costfunction='OLS', measurementId=0, yearEnd=None, lhs=True):
        """
        Read the relative costfunction values for all parameter sets for a given model and time step
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0
        assert yearEnd is None or type(yearEnd) is int and 0 <= yearEnd
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim_c.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT c.simulationId AS simulationId, (c.{}-d.{})/d.{} AS relError FROM CostfuctionEvaluation AS c, CostfuctionEvaluation AS d, Simulation AS sim_c, Simulation AS sim_d, Convergence AS con_c, Convergence AS con_d WHERE sim_c.model = ? AND sim_c.timestep = ?{:s} AND c.year = ? AND c.measurementId = ? AND sim_c.simulationId = c.simulationId AND sim_c.simulationId = con_c.simulationId AND con_c.convergence = ? AND sim_d.timestep = ? AND sim_c.model = sim_d.model AND sim_c.parameterId = sim_d.parameterId AND sim_c.concentrationId = sim_d.concentrationId AND d.simulationId = sim_d.simulationId AND d.measurementId = c.measurementId AND d.year = {:s} AND sim_d.simulationId = con_d.simulationId AND con_d.convergence = ? ORDER BY c.simulationId;'.format(costfunction, costfunction, costfunction, parameterStr, 'c.year' if yearEnd is not None else '{:d}'.format(yearEnd))
        self._c.execute(sqlcommand,  (model, timestep, year, measurementId, int(True), 1, int(True)))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1

        return simdata


    def read_costfunction_avg_error_for_model(self, model, year=10000, costfunction='OLS', measurementId=0, yearEnd=None, lhs=True):
        """
        Read the average of the relative costfunction value error over all parameter sets of the latin hypercube sample for the given model and year.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0
        assert yearEnd is None or type(yearEnd) is int and 0 <= yearEnd
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim_c.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim_c.timestep as timestep, AVG(ABS((c.{}-d.{})/d.{})) AS relError FROM CostfuctionEvaluation AS c, CostfuctionEvaluation AS d, Simulation AS sim_c, Simulation AS sim_d, Convergence AS con_c, Convergence AS con_d WHERE sim_c.model = ? AND c.year = ? AND c.measurementId = ?{:s} AND sim_c.simulationId = c.simulationId AND sim_c.simulationId = con_c.simulationId AND con_c.convergence = ? AND sim_d.timestep = ? AND sim_c.model = sim_d.model AND sim_c.parameterId = sim_d.parameterId AND sim_c.concentrationId = sim_d.concentrationId AND sim_d.simulationId = con_d.simulationId AND con_d.convergence = ? AND d.simulationId = sim_d.simulationId AND d.measurementId = c.measurementId AND d.year = {:s} GROUP BY sim_c.timestep;'.format(costfunction, costfunction, costfunction, parameterStr, 'c.year' if yearEnd is not None else '{:d}'.format(yearEnd))
        self._c.execute(sqlcommand,  (model, year, measurementId, int(True), 1, int(True)))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    def read_rel_costfunction_for_model_timestep(self, model, timestep, year=10000, costfunction='OLS', measurement=0, lhs=True):
        """
        Read for every parameterId the costfunction value for the given model, time step and year from the database.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.simulationId AS simid, (cost.{}-cost_ref.{})/cost_ref.{} AS Error FROM Simulation AS sim, Simulation AS sim_ref, CostfuctionEvaluation AS cost, CostfuctionEvaluation AS cost_ref, Convergence AS con WHERE sim.model = ? AND sim.timestep = ?{} AND sim.simulationId = con.simulationId AND con.convergence = ? AND sim.simulationId = cost.simulationId AND cost.year = ? AND cost.measurementId = ? AND sim.model = sim_ref.model AND sim_ref.timestep = 1 AND sim.parameterId = sim_ref.parameterId AND sim.concentrationId = sim_ref.concentrationId AND sim_ref.simulationId = cost_ref.simulationId AND cost_ref.year = cost.year AND cost_ref.measurementId = cost.measurementId ORDER BY sim.simulationId'.format(costfunction, costfunction, costfunction, parameterStr)
        self._c.execute(sqlcommand,  (model, timestep, int(True), year, measurement))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows)))

        i = 0
        for row in simrows:
            simdata[i] = row[1]
            i = i+1

        return simdata


    def get_table_norm_error(self, model='N', parameterId=0, year=10000):
        """
        Get the norm values for the four different norms for all time steps for the given model, parameterId and year
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert type(year) is int and year >= 0

        sqlcommand = 'SELECT sim.timestep AS timestep, tracer_2.tracer_1dt/tracer_2_ref.tracer as Norm_2, tracer_box.tracer_1dt/tracer_box_ref.tracer AS Norm_box, tracer_tra.tracer_1dt/tracer_tra_ref.tracer as Norm_tra FROM Simulation AS sim, Tracer2Norm AS tracer_2_ref, TracerDifference2Norm as tracer_2, TracerBoxweightedNorm AS tracer_box_ref, TracerDifferenceBoxweightedNorm AS tracer_box, TracerTrajectory2Norm AS tracer_tra_ref, TracerDifferenceTrajectory2Norm AS tracer_tra, Convergence AS con WHERE sim.parameterId = ? AND sim.model = ? AND sim.timestep > ? AND sim.simulationId = con.simulationId AND con.convergence = ? AND tracer_2.simulationId = sim.simulationId AND tracer_2.year = ? AND tracer_2.simulationId = tracer_2_ref.simulationId AND tracer_2.year = tracer_2_ref.year AND tracer_box.simulationId = sim.simulationId AND tracer_box.year = ? AND tracer_box.simulationId = tracer_box_ref.simulationId AND tracer_box.year = tracer_box_ref.year AND tracer_tra.simulationId = sim.simulationId AND tracer_tra.year = ? AND tracer_tra.simulationId = tracer_tra_ref.simulationId AND tracer_tra.year = tracer_tra_ref.year ORDER BY sim.timestep'
        self._c.execute(sqlcommand, (parameterId, model, 1, int(True), year, year, year))
        simrows = self._c.fetchall()
        norms = np.empty(shape=(len(simrows), 4))

        i = 0
        for row in simrows:
            norms[i,:] = np.array([row[0], row[1], row[2], row[3]])
            i = i+1
        return norms


    def get_table_norm_value(self, parameterId=0, model='N', timestep=1, year=10000, norm='2'):
        """
        Get the norm value for the given values
        @author: Markus Pfeil
        """
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM

        sqlcommand = 'SELECT tracer_2.tracer_1dt/tracer.tracer as Norm FROM tracerDifference{}Norm AS tracer_2, Tracer{}Norm AS tracer, Simulation AS sim, Convergence AS con WHERE sim.parameterId = ? AND sim.model = ? AND sim.timestep = ? AND sim.simulationId = con.simulationId AND con.convergence = ? AND sim.simulationId = tracer_2.simulationId AND sim.simulationId = tracer.simulationId AND tracer.year = ? AND tracer_2.year = ?'.format(norm, norm)
        self._c.execute(sqlcommand, (parameterId, model, timestep, int(True), year, year))
        normValue = self._c.fetchall()
        assert(len(normValue)) in [0, 1]
        return normValue


    def get_table_costfunction_value(self, parameterId=0, model='N', timestep=1, year=10000, costfunction='OLS', measurementId=0):
        """
        Get the costfunction value for the given values
        @author: Markus Pfeil
        """
        assert type(parameterId) is int and parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert type(measurementId) is int and measurementId >= 0

        sqlcommand = 'SELECT sim.timestep AS timestep, c.{} AS {} FROM CostfuctionEvaluation AS c, Simulation AS sim, Convergence AS con WHERE sim.model = ? AND sim.timestep = ? AND sim.parameterId = ? AND sim.simulationId = con.simulationId AND con.convergence = ? AND c.year = ? AND c.measurementId = ? AND sim.simulationId = c.simulationId ORDER BY c.simulationId;'.format(costfunction, costfunction)
        self._c.execute(sqlcommand, (model, timestep, parameterId, int(True), year, measurementId))
        costfunctionValue = self._c.fetchall()
        assert(len(costfunctionValue)) in [0, 1]
        return costfunctionValue


    def get_table_convergence(self, model='N', convergence=False, lhs=True):
        """
        Get the count of parameter sets with/without spin-up convergence
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(convergence) is bool
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.timestep AS Timestep, COUNT(*) AS Count FROM Simulation AS sim, Convergence AS con WHERE sim.model = ?{} AND sim.simulationId = con.simulationId AND con.convergence = ? GROUP BY sim.timestep;'.format(parameterStr)
        self._c.execute(sqlcommand, (model, int(convergence)))
        simrows = self._c.fetchall()
        assert len(simrows) in range(0,8)
        convergenceValue = np.empty(shape=(len(simrows), 2))
        i = 0
        for row in simrows:
            convergenceValue[i,:] = np.array([row[0], row[1]])
            i = i+1
        return convergenceValue


    def get_table_spinup_tolerance(self, model='N', tolerance=10**(-4), lhs=True):
        """
        Get the count of parameter sets which reached the given tolerance for the spin-up
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(tolerance) is float and tolerance >= 0
        assert type(lhs) is bool

        if lhs:
            parameterStr = ' AND sim.parameterId > 0'
        else:
            parameterStr = ''

        sqlcommand = 'SELECT sim.timestep AS timestep, count(*) FROM Spinup AS sp, Simulation AS sim, Convergence AS con WHERE sim.model = ?{} AND sim.simulationId = sp.simulationId AND sp.tolerance <= ? AND sim.simulationId = con.simulationId AND con.convergence = ? AND NOT EXISTS ( SELECT * FROM Spinup AS sp1 WHERE sp1.simulationId = sp.simulationId AND sp1.tolerance <= ? AND sp1.year < sp.year) GROUP BY sim.timestep;'.format(parameterStr)
        self._c.execute(sqlcommand, (model, tolerance, int(True), tolerance))
        simrows = self._c.fetchall()
        assert len(simrows) in range(0,8)
        spinupToleranceValue = np.empty(shape=(len(simrows), 2))
        i = 0
        for row in simrows:
            spinupToleranceValue[i,:] = np.array([row[0], row[1]])
            i = i+1
        return spinupToleranceValue

