#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import abstractmethod
import logging
import numpy as np
import os
import sqlite3
import time

import metos3dutil.database.constants as DB_Constants
from metos3dutil.database.AbstractClassDatabaseMetos3d import AbstractClassDatabaseMetos3d
import metos3dutil.metos3d.constants as Metos3d_Constants


class DatabaseMetos3d(AbstractClassDatabaseMetos3d):
    """
    Abstract class for the database access
    """

    def __init__(self, dbpath, completeTable=True):
        """
        Initialization of the database connection

        Parameters
        ----------
        dbpath : str
            Path to the sqlite file of the sqlite database
        completeTable : bool
            If True, use each column of a database table in sql queries

        Raises
        ------
        AssertionError
            If the dbpath does not exists.
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        AbstractClassDatabaseMetos3d.__init__(self, dbpath)

        self._completeTable = completeTable


    def _create_table_initialConcentration(self):
        """
        Create table InitialConcentration
        """
        self._c.execute('''CREATE TABLE InitialConcentration (concentrationId INTEGER NOT NULL, concentrationTyp TEXT NOT NULL, distribution TEXT, tracerDistribution TEXT, differentTracer INTEGER, N TEXT NOT NULL, P TEXT, Z TEXT, D TEXT, DOP TEXT, PRIMARY KEY (concentrationId))''')


    def get_concentrationId_constantValues(self, metos3dModel, concentrationValues):
        """
        Returns concentrationId for constant initial values

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationValues : list [float]
            Constant tracer concentration for each tracer of the metos3dModel

        Returns
        -------
        int
            concentrationId of the constant tracer concentration

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(concentrationValues) is list

        concentrationParameter = Metos3d_Constants.TRACER_MASK[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[metos3dModel]]
        assert len(concentrationParameter) == len(concentrationValues)

        sqlcommand = "SELECT concentrationId FROM InitialConcentration WHERE concentration_typ = 'constant' AND {} = ?".format(concentrationParameter[0])
        for i in range(1, len(concentrationValues)):
            sqlcommand = sqlcommand + ' AND {} = ?'.format(concentrationParameter[i])
        concentrationenParameterNone = Metos3d_Constants.TRACER_MASK[np.invert(Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[metos3dModel])]
        if self._completeTable:
            for i in range(0, len(concentrationenParameterNone)):
                sqlcommand = sqlcommand + ' AND {} IS NULL'.format(concentrationenParameterNone[i])
        self._c.execute(sqlcommand, tuple(concentrationValues))
        concentrationId = self._c.fetchall()
        assert len(concentrationId) == 1
        return concentrationId[0][0]


    def get_concentrationId_vectorValues(self, metos3dModel, tracerNum=0, distribution='Lognormal', tracerDistribution='set_mass'):
        """
        Returns concentrationId for vector initial values

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        tracerNum : int, default: 0
            Number of the tracer concentration vector
        distribution : {'Lognormal', 'Normal', 'OneBox', 'Uniform'},
            default: 'Lognormal'
            Distribtution used to generate the tracer concentratin
        tracerDistribution : {'set_mass', 'random_mass'}, default: 'set_mass'
            #TODO

        Returns
        -------
        int
            concentrationId of the constant tracer concentration

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert 0 <= tracerNum and tracerNum < 100
        assert distribution in ['Lognormal', 'Normal', 'OneBox', 'Uniform']
        assert tracerDistribution in ['set_mass', 'random_mass']

        concentrationParameter = Metos3d_Constants.TRACER_MASK[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[metos3dModel]]
        sqlcommand = 'SELECT concentrationId FROM InitialConcentration WHERE concentrationTyp = ? AND distribution = ? AND tracerDistribution = ? AND differentTracer = ?'
        for i in range(0, len(concentrationParameter)):
            sqlcommand = sqlcommand + " AND {} = 'InitialValue_Tracer_{}_{:0>3}.petsc'".format(concentrationParameter[i], i, tracerNum)
        self._c.execute(sqlcommand, ('vector', distribution, tracerDistribution, self.DifferentTracer[metos3dModel])) #TODO Reset DifferentTracer
        concentrationId = self._c.fetchall()
        assert len(concentrationId) == 1
        return concentrationId[0][0]


    def _create_table_simulation(self):
        """
        Create table Simulation
        """
        self._c.execute('''CREATE TABLE Simulation (simulationId INTEGER NOT NULL, model TEXT NOT NULL, parameterId INTEGER NOT NULL REFERENCES Parameter(parameterId), concentrationId INTEGER NOT NULL REFERENCES InitialConcentration(concentrationID), timestep INTEGER NOT NULL, PRIMARY KEY (simulationId))''')


    @abstractmethod
    def get_simulationId(self, metos3dModel, parameterId, concentrationId):
        """
        Returns the simulationId

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration

        Returns
        -------
        int
            simulationId for the combination of model, parameterId and
            concentrationId

        Raises
        ------
        AssertionError
            If no entry for the model, parameterId, concentrationId and
            timestep exists in the database table Simulation
        """
        pass


    def _create_table_spinup(self):
        """
        Create table Spinup
        """
        c.execute('''CREATE TABLE Spinup (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, tolerance REAL NOT NULL, spinupNorm REAL, PRIMARY KEY (simulationId, year))''')


    def get_spinup_year_for_tolerance(self, simulationId, tolerance=10**(-4)):
        """
        Returns the first model year of the spin up with less tolerance

        Returns the model year of the spin up calculation where the tolerance
        fall below the given tolerance value. If the tolerance of the spin up
        is higher than the given tolerance for every model year, return None.

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        tolerance : float, default: 0.0001
            Tolerance value for the spin up norm

        Returns
        -------
        None or int
            If the spin up norm is always greater than the given tolerance,
            return None. Otherwise, the model year in which the spin up norm
            falls below the tolerance for the first time is returned.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(tolerance) is float and tolerance > 0

        sqlcommand = 'SELECT sp.year FROM Spinup AS sp WHERE sp.simulationId = ? AND sp.tolerance < ? AND NOT EXISTS (SELECT * FROM Spinup AS sp1 WHERE sp1.simulationId = sp.simulationId AND sp1.tolerance < ? AND sp1.year < sp.year)'
        self._c.execute(sqlcommand, (simulationId, tolerance, tolerance))
        count = self._c.fetchall()
        assert len(count) == 1 or len(count) == 0
        if len(count) == 1:
            return count[0][0] + 1
        else:
            return None


    def read_spinup_values_for_simid(self, simulationId):
        """
        Returns the spin up norm values of a simulation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation

        Returns
        -------
        numpy.ndarray
            2D array with the year and the tolerance
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'SELECT year, tolerance FROM Spinup WHERE simulationId = ? ORDER BY year;'
        self._c.execute(sqlcommand, (simulationId, ))
        simrows = self._c.fetchall()
        simdata = np.empty(shape=(len(simrows), 2))

        i = 0
        for row in simrows:
            simdata[i,:] = np.array([row[0], row[1]])
            i = i+1
        return simdata


    @abstractmethod
    def read_spinup_tolerance(self, metos3dModel, concentrationId, year):
        """
        Returns the spin up tolerance for all parameterIds

        Returns the spin up tolerance of all simulations using the given model
        and concentrationId for the given model year.

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationId : int
            Id of the concentration
        year : int
            Model year of the spin up calculation

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId and the tolerance
        """
        pass


    @abstractmethod
    def read_spinup_year(self, model, concentrationId):
        """
        Returns the required years to reach the given spin up tolerance

        Returns the required model years to reach the given spin up tolerance
        for every parameterId.

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationId : int
            Id of the concentration

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId and the required model year
        """
        pass


    def check_spinup(self, simulationId, expectedCount):
        """
        Check the number of spin up norm entries for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM Spinup WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1
        return count[0][0] == expectedCount


    def insert_spinup(self, simulationId, year, tolerance, spinupNorm, overwrite=False):
        """
        Insert spin up value

        Insert spin up value. If a spin up database entry for the simulationId
        and year already exists, the existing entry is deleted and the new one
        is inserted (if the flag overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        year : int
            Model year of the spin up calculation
        tolerance : float
            Tolerance of the spin up norm
        spinupNorm : float
            Spin up Norm value
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
        assert type(tolerance) is float and 0 <= tolerance
        assert type(spinupNorm) is float and 0 <= spinupNorm
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year FROM Spinup WHERE simulationId = ? AND year = ?'
        self._c.execute(sqlcommand_select, (simulationId, year))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Spinup WHERE simulationId = ? AND year = ?'
                self._c.execute(sqlcommand, (simulationId, year))
        else:
            assert len(dataset) == 0

        #Generate and insert spin-up value
        purchases = []
        purchases.append((simulationId, year, tolerance, spinupNorm))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO Spinup VALUES (?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_spinup(self, simulationId):
        """
        Delete entries of the spin up calculation

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'DELETE FROM Spinup WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId,))
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
        self._c.execute('''CREATE TABLE Tracer{:s}{:s}Norm (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, tracer REAL NOT NULL, N REAL NOT NULL, DOP REAL, P REAL, Z REAL, D REAL, PRIMARY KEY (simulationId, year))'''.format(trajectory, norm))


    def check_tracer_norm(self, simulationId, expectedCount, norm='2', trajectory=''):
        """
        Check the number of tracer norm entries for the given simulationId

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up
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
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT COUNT(*) FROM Tracer{}{}Norm WHERE simulationId = ?'.format(trajectory, norm)
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerNorm: Expected: {} Get: {} (Norm: {})***'.format(expectedCount, count[0][0], norm))

        return count[0][0] == expectedCount


    def insert_tracer_norm_tuple(self, simulationId, year, tracer, N, DOP=None, P=None, Z=None, D=None, norm='2', trajectory='', overwrite=False):
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
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?'.format(trajectory, norm)
        self._c.execute(sqlcommand_select, (simulationId, year))
        dataset = self._c.fetchall()
        if overwrite:
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?'.format(trajectory, norm)
                self._c.execute(sqlcommand, (simulationId, year))
        else:
            assert len(dataset) == 0

        #Generate insert for the tracer norm
        purchases = []
        purchases.append((simulationId, year, tracer, N, DOP, P, Z, D))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO Tracer{}{}Norm VALUES (?,?,?,?,?,?,?,?)'.format(trajectory, norm), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_tracer_norm(self, simulationId, norm='2', trajectory=''):
        """
        Delete entries of the tracer norm

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        """
        assert type(simulationId) is int and simulationId >= 0
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'DELETE FROM Tracer{}{}Norm WHERE simulationId = ?'.format(trajectory, norm) 
        self._c.execute(sqlcommand, (simulationId,))
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
        self._c.execute('''CREATE TABLE TracerDifference{:s}{:s}Norm (simulationIdA INTEGER NOT NULL REFERENCES Simulation(simulationId), simulationIdB INTEGER NOT NULL REFERENCES Simulation(simulationId), yearA INTEGER NOT NULL, yearB INTEGER NOT NULL, tracer REAL NOT NULL, N REAL NOT NULL, DOP REAL, P REAL, Z REAL, D REAL, PRIMARY KEY (simulationIdA, simulationIdB, yearA, yearB))'''.format(trajectory, norm))


    def check_difference_tracer_norm(self, simulationIdA, simulationIdB, expectedCount, norm='2', trajectory=''):
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
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'SELECT COUNT(*) FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ?'.format(trajectory, norm)
        self._c.execute(sqlcommand, (simulationIdA, simulationIdB))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerDifferenceNorm: Expected: {} Get: {} (Norm: {})***'.format(expectedCount, count[0][0], norm))

        return count[0][0] == expectedCount


    def insert_difference_tracer_norm_tuple(self, simulationIdA, simulationIdB, yearA, yearB, tracerDifferenceNorm, N, DOP=None, P=None, Z=None, D=None, norm='2', trajectory='', overwrite=False):
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
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationIdA, simulationIdB, yearA, yearB FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?'.format(trajectory, norm)
        if overwrite:
            #Test, if dataset for this simulationIdA, simulationIdB, yearA and yearB combination exists
            self._c.execute(sqlcommand_select, (simulationIdA, simulationIdB, yearA, yearB))
            dataset = self._c.fetchall()
            #Remove database entry for this simulationIdA, simulationIdB, yearA and yearB
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM TracerDifference{}{}Norm WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?'.format(trajectory, norm)
                self._c.execute(sqlcommand, (simulationIdA, simulationIdB, yearA, yearB))
        else:
            self._c.execute(sqlcommand_select, (simulationIdA, simulationIdB, yearA, yearB))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate insert for the tracer norm
        purchases = []
        purchases.append((simulationIdA, simulationIdB, yearA, yearB, tracerDifferenceNorm, N, DOP, P, Z, D))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO TracerDifference{}{}Norm VALUES (?,?,?,?,?,?,?,?,?,?)'.format(trajectory, norm), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_difference_tracer_norm(self, simulationIdA, norm='2', trajectory=''):
        """
        Delete entries of the norm between two tracers

        Delete the entries of the norm between two tracers where the first
        tracer is identified by the given simulationId.

        Parameters
        ----------
        simulationIdA : int
            Id defining the parameter for spin up calculation
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory
        """
        assert type(simulationIdA) is int and simulationIdA >= 0
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        sqlcommand = 'DELETE FROM TracerDifference{}{}Norm WHERE simulationIdA = ?'.format(trajectory, norm)
        self._c.execute(sqlcommand, (simulationIdA,))
        self._conn.commit()


    @abstractmethod
    def read_rel_norm(self, metos3dModel, concentrationId, year=None, norm='2', parameterId=None, trajectory=''):
        """
        Returns the relative error

        Returns the relative error of all simulations using the given model
        and concentrationId. If parameterId is not None, this function returns
        only the relative difference for the given parameterId. If the year is
        not None, this function returns the relative error for the given year.

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationId : int
            Id of the concentration
        year : None or int, default: None
            Model year to return the relative error. If None, return the
            relative error for the last model year of the simulation.
        norm : {'2', 'Boxweighted', 'BoxweightedVol'}, default: '2'
            Used norm
        parameterId : None or int, default: None
            Id of the parameter of the latin hypercube example. If None, this
            function returns the relative for all parameterIds.
        trajectory : {'', 'Trajectory'}, default: ''
            Norm over the whole trajectory

        Returns
        -------
        numpy.ndarray
            2D array with the simulationId and the relative error
        """
        pass

