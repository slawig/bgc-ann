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
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


class DatabaseMetos3d(AbstractClassDatabaseMetos3d):
    """
    Abstract class for the database access
    """

    def __init__(self, dbpath, completeTable=True, createDb=False):
        """
        Initialization of the database connection

        Parameters
        ----------
        dbpath : str
            Path to the sqlite file of the sqlite database
        completeTable : bool
            If True, use each column of a database table in sql queries
        createDb : bool, default: False
            If True, the database does not have to exist and can be created
            using the function create_database

        Attributes
        ----------
        _completeTable : bool
            If True, use each column of a database table in sql queries
        _deviationNegativeValues : bool, default: False
            If True, uses in table deviation negative values for each tracer

        Raises
        ------
        AssertionError
            If the dbpath does not exists.
        """
        assert type(createDb) is bool
        assert createDb or os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        AbstractClassDatabaseMetos3d.__init__(self, dbpath, createDb=createDb)

        self._completeTable = completeTable
        self._deviationNegativeValues = False


    def set_deviationNegativeValues(self, deviationNegativeValues):
        """
        Sets the use of negative values in the table deviation

        Parameters
        ----------
        deviationNegativeValues : bool
            If True, use deviation of negative values for each tracer
        """
        assert type(deviationNegativeValues) is bool

        self._deviationNegativeValues = deviationNegativeValues


    def _create_table_initialConcentration(self):
        """
        Create table InitialConcentration
        """
        self._c.execute('''CREATE TABLE InitialConcentration (concentrationId INTEGER NOT NULL, concentrationTyp TEXT NOT NULL, N TEXT NOT NULL, P TEXT, Z TEXT, D TEXT, DOP TEXT, UNIQUE (concentrationTyp, N, P, Z, D, DOP), PRIMARY KEY (concentrationId))''')


    def _init_table_initialConcentration(self):
        """
        Initial insert of initial concentration data sets
        """
        purchases = []
        concentrationId = 0

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS[:-1]:
            concentration = [Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][0]] + Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][1:-1]
            while len(concentration) < len(Metos3d_Constants.TRACER_MASK)-1:
                concentration.append(None)
            concentration.append(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][-1] if len(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel]) > 1 else None)

            purchases.append((concentrationId, 'constant') + tuple(concentration))
            concentrationId += 1

        self._c.executemany('INSERT INTO InitialConcentration VALUES (?,?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def exists_initialConcentration(self, N, P=None, Z=None, D=None, DOP=None):
        """
        Returns if a database entry exists for the initial concentration

        Parameters
        ----------
        N : str or float
            If concentrationTyp is 'constant', global mean concentration of
            the N tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the N tracer.
        P : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the P tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the P tracer.
        Z : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the Z tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the Z tracer.
        D : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the D tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the D tracer.
        DOP : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the DOP tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the DOP tracer.

        Returns
        -------
        bool
            True if an entry exists for the given concentration
        """
        assert (type(N) is float and (P is None or type(P) is float) and (Z is None or type(Z) is float) and (D is None or type(D) is float) and (DOP is None or type(DOP) is float)) or (type(N) is str and (P is None or type(P) is str) and (Z is None or type(Z) is str) and (D is None or type(D) is str) and (DOP is None or type(DOP) is str))

        sqlcommand = 'SELECT concentrationId FROM InitialConcentration WHERE concentrationTyp = ? AND N = ?'
        sqltuple = ('constant' if type(N) is float else 'vector',  N)

        if P is None:
            sqlcommand = sqlcommand + ' AND P IS NULL'
        else:
            sqlcommand = sqlcommand + ' AND P = ?'
            sqltupel = sqltupel + (P,)

        if Z is None:
            sqlcommand = sqlcommand + ' AND Z IS NULL'
        else:
            sqlcommand = sqlcommand + ' AND Z = ?'
            sqltupel = sqltupel + (Z,)

        if D is None:
            sqlcommand = sqlcommand + ' AND D IS NULL'
        else:
            sqlcommand = sqlcommand + ' AND D = ?'
            sqltupel = sqltupel + (D,)

        if DOP is None:
            sqlcommand = sqlcommand + ' AND DOP IS NULL'
        else:
            sqlcommand = sqlcommand + ' AND DOP = ?'
            sqltupel = sqltupel + (DOP,)

        self._c.execute(sqlcommand, sqltupel)
        concentrationId = self._c.fetchall()
        return len(concentrationId) > 0


    def get_concentration(self, concentrationId):
        """
        Returns the concentration for the given concentrationId

        Parameters
        ----------
        concentrationId : int
            Id of the initial tracer concentration

        Returns
        -------
        numpy.ndarray
            Numpy array with the initial concentration for all tracers

        Raises
        ------
        AssertionError
            If no database entry exists for the concentrationId
        """
        assert type(concentrationId) is int and 0 <= concentrationId

        sqlcommand = 'SELECT N, P, Z, D, DOP FROM InitialConcentration WHERE concentrationId = ?'
        self._c.execute(sqlcommand, (concentrationId, ))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return np.array(dataset[0])


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

        sqlcommand = "SELECT concentrationId FROM InitialConcentration WHERE concentrationTyp = 'constant' AND {} = ?".format(concentrationParameter[0])
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


    def insert_initialConcentration(self, concentrationTyp, N, P=None, Z=None, D=None, DOP=None):
        """
        Insert initial concentration values

        Parameters
        ----------
        concentrationTyp : {'vector', 'constant'}
            Use constant initial concentration or an initial concentration
            defined with vectors
        N : str or float
            If concentrationTyp is 'constant', global mean concentration of
            the N tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the N tracer.
        P : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the P tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the P tracer.
        Z : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the Z tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the Z tracer.
        D : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the D tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the D tracer.
        DOP : str or float or None, default: None
            If concentrationTyp is 'constant', global mean concentration of
            the DOP tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the DOP tracer.

        Raises
        ------
        sqlite3.OperationalError
            If the initial concentration could not be successfully inserted
            into the database after serveral attempts

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try
        """
        assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
        assert concentrationTyp == 'constant' and type(N) is float or concentrationTyp == 'vector' and type(N) is str
        assert P is None or concentrationTyp == 'constant' and type(P) is float or concentrationTyp == 'vector' and type(P) is str
        assert Z is None or concentrationTyp == 'constant' and type(Z) is float or concentrationTyp == 'vector' and type(Z) is str
        assert D is None or concentrationTyp == 'constant' and type(D) is float or concentrationTyp == 'vector' and type(D) is str
        assert DOP is None or concentrationTyp == 'constant' and type(DOP) is float or concentrationTyp == 'vector' and type(DOP) is str

        if not self.exists_initialConcentration(N, P=P, Z=Z, D=D, DOP=DOP):
            #Insert initial concentration into the database
            sqlcommand = 'SELECT MAX(concentrationId) FROM InitialConcentration'
            self._c.execute(sqlcommand)
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            concentrationId = dataset[0][0] + 1

            purchases = [(concentrationId, concentrationTyp, N, P, Z, D, DOP)]

            inserted = False
            insertCount = 0
            while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
                try:
                    self._c.executemany('INSERT INTO InitialConcentration VALUES (?,?,?,?,?,?,?)', purchases)
                    self._conn.commit()
                    inserted = True
                except sqlite3.OperationalError:
                    insertCount += 1
                    #Wait for the next insert
                    time.sleep(DB_Constants.TIME_SLEEP)


    def _create_table_simulation(self):
        """
        Create table Simulation
        """
        self._c.execute('''CREATE TABLE Simulation (simulationId INTEGER NOT NULL, model TEXT NOT NULL, parameterId INTEGER NOT NULL REFERENCES Parameter(parameterId), concentrationId INTEGER NOT NULL REFERENCES InitialConcentration(concentrationId), timestep INTEGER NOT NULL, UNIQUE (model, parameterId, concentrationId, timestep), PRIMARY KEY (simulationId))''')


    def exists_simulaiton(self, metos3dModel, parameterId, concentrationId, timestep=1):
        """
        Returns if a simulation entry exists for the given values

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation

        Returns
        -------
        bool
            True if an entry exists for the given values
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND timestep = ?'
        self._c.execute(sqlcommand, (metos3dModel, parameterId, concentrationId, timestep))
        simulationId = self._c.fetchall()
        return len(simulationId) > 0


    def get_simulationId(self, metos3dModel, parameterId, concentrationId, timestep=1):
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
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation

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
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND timestep = ?'
        self._c.execute(sqlcommand, (metos3dModel, parameterId, concentrationId, timestep))
        simulationId = self._c.fetchall()
        assert len(simulationId) == 1
        return simulationId[0][0]


    def insert_simulation(self, metos3dModel, parameterId, concentrationId, timestep=1):
        """
        Insert simulation data set

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationId : int
            Id of the concentration
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation

        Returns
        -------
        int
            simulationId for the combination of model, parameterId and
            concentrationId

        Raises
        ------
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and 0 <= concentrationId
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        if self.exists_simulaiton(metos3dModel, parameterId, concentrationId, timestep=timestep):
            #Simulation already exists in the database
            simulationId = self.get_simulationId(metos3dModel, parameterId, concentrationId, timestep=timestep)
        else:
            #Insert simulation into the database
            sqlcommand = 'SELECT MAX(simulationId) FROM Simulation'
            self._c.execute(sqlcommand)
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            simulationId = dataset[0][0] + 1

            purchases = [(simulationId, metos3dModel, parameterId, concentrationId, timestep)]
            inserted = False
            insertCount = 0
            while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
                try:
                    self._c.executemany('INSERT INTO Simulation VALUES (?,?,?,?,?)', purchases)
                    self._conn.commit()
                    inserted = True
                except sqlite3.OperationalError:
                    insertCount += 1
                    #Wait for the next insert
                    time.sleep(DB_Constants.TIME_SLEEP)

        return simulationId


    def _create_table_spinup(self):
        """
        Create table Spinup
        """
        self._c.execute('''CREATE TABLE Spinup (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, tolerance REAL NOT NULL, spinupNorm REAL, PRIMARY KEY (simulationId, year))''')


    def get_spinup_year_for_tolerance(self, simulationId, tolerance=0.0001):
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
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

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
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

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


    def _create_table_DeviationTracer(self):
        """
        Create table DeviationTracer

        Notes
        -----
        If attribute self._deviationNegativeValues is True, create two columns
        for the number of boxes with negative concentrations and the sum of
        the negative concentrations for each tracer.
        """
        if self._deviationNegativeValues:
            self._c.execute('''CREATE TABLE DeviationTracer (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, N_mean REAL NOT NULL, N_var REAL NOT NULL, N_min REAL NOT NULL, N_max REAL NOT NULL, N_negative_count INTEGER NOT NULL, N_negative_sum REAL NOT NULL, DOP_mean REAL, DOP_var REAL, DOP_min REAL, DOP_max REAL, DOP_negative_count INTEGER, DOP_negative_sum REAL, P_mean REAL, P_var REAL, P_min REAL, P_max REAL, P_negative_count INTEGER, P_negative_sum REAL, Z_mean REAL, Z_var REAL, Z_min REAL, Z_max REAL, Z_negative_count INTEGER, Z_negative_sum REAL, D_mean REAL, D_var REAL, D_min REAL, D_max REAL, D_negative_count INTEGER, D_negative_sum REAL, PRIMARY KEY (simulationId, year))''')
        else:
            self._c.execute('''CREATE TABLE DeviationTracer (simulationId INTEGER NOT NULL REFERENCES Simulation(simulationId), year INTEGER NOT NULL, N_mean REAL NOT NULL, N_var REAL NOT NULL, N_min REAL NOT NULL, N_max REAL NOT NULL, DOP_mean REAL, DOP_var REAL, DOP_min REAL, DOP_max REAL, P_mean REAL, P_var REAL, P_min REAL, P_max REAL, Z_mean REAL, Z_var REAL, Z_min REAL, Z_max REAL, D_mean REAL, D_var REAL, D_min REAL, D_max REAL, PRIMARY KEY (simulationId, year))''')


    def check_tracer_deviation(self, simulationId, expectedCount):
        """
        Check the number of entries of the deviation values for simulationId

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

        sqlcommand = 'SELECT COUNT(*) FROM DeviationTracer WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***Check DeviationTracer: Expected: {} Get: {}***'.format(expectedCount, count[0][0]))

        return count[0][0] == expectedCount


    def insert_deviation_tracer_tuple(self, simulationId, year, N_mean, N_var, N_min, N_max, N_negative_count=None, N_negative_sum=None, DOP_mean=None, DOP_var=None, DOP_min=None, DOP_max=None, DOP_negative_count=None, DOP_negative_sum=None, P_mean=None, P_var=None, P_min=None, P_max=None, P_negative_count=None, P_negative_sum=None, Z_mean=None, Z_var=None, Z_min=None, Z_max=None, Z_negative_count=None, Z_negative_sum=None, D_mean=None, D_var=None, D_min=None, D_max=None, D_negative_count=None, D_negative_sum=None, overwrite=False):
        """
        Insert deviation for a tracer

        Insert deviation for a tracer. If a database entry for the deviation
        for this simulationId and year already exists, the existing entry is
        deleted and the new one is inserted (if the flag overwrite is True).

        Parameters
        ----------
        simulationId : int
            Id defining the parameter for spin up calculation of the used
            tracer
        year : int
            Model year of the spin up calculation for the tracer
        N_mean : float
            Mean of the N tracer
        N_var : float
            Variance of the N tracer
        N_min : float
            Minimum of the N tracer
        N_max : float
            Maximum of the N tracer
        N_negative_count : int or None, default: None
            Number of boxes for which the concentration of the N tracer is
            negative
        N_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the N tracer
        DOP_mean : float or None, default: None
            Mean of the DOP tracer
        DOP_var : float or None, default: None
            Variance of the DOP tracer
        DOP_min : float or None, default: None
            Minimum of the DOP tracer
        DOP_max : float or None, default: None
            Maximum of the DOP tracer
        DOP_negative_count : int or None, default: None
            Number of boxes for which the concentration of the DOP tracer is
            negative
        DOP_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the DOP tracer
        P_mean : float or None, default: None
            Mean of the P tracer
        P_var : float or None, default: None
            Variance of the P tracer
        P_min : float or None, default: None
            Minimum of the P tracer
        P_max : float or None, default: None
            Maximum of the P tracer
        P_negative_count : int or None, default: None
            Number of boxes for which the concentration of the P tracer is
            negative
        P_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the P tracer
        Z_mean : float or None, default: None
            Mean of the Z tracer
        Z_var : float or None, default: None
            Variance of the Z tracer
        Z_min : float or None, default: None
            Minimum of the Z tracer
        Z_max : float or None, default: None
            Maximum of the Z tracer
        Z_negative_count : int or None, default: None
            Number of boxes for which the concentration of the Z tracer is
            negative
        Z_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the Z tracer
        D_mean : float or None, default: None
            Mean of the D tracer
        D_var : float or None, default: None
            Variance of the D tracer
        D_min : float or None, default: None
            Minimum of the D tracer
        D_max : float or None, default: None
            Maximum of the D tracer
        D_negative_count : int or None, default: None
            Number of boxes for which the concentration of the D tracer is
            negative
        D_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the D tracer
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
        If attribute self._deviationNegativeValues is True, insert values for
        the number of boxes with negative concentrations and the sum of the
        negative concentrations for each tracer.
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(N_mean) is float and type(N_var) is float and type(N_min) is float and type(N_max) is float and ((self._deviationNegativeValues and type(N_negative_count) is int and type(N_negative_sum) is float) or (not self._deviationNegativeValues))
        assert (DOP_mean is None and DOP_var is None and DOP_min is None and DOP_max is None and DOP_negative_count is None and DOP_negative_sum is None) or (type(DOP_mean) is float and type(DOP_var) is float and type(DOP_min) is float and type(DOP_max) is float and ((self._deviationNegativeValues and type(DOP_negative_count) is int and type(DOP_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert (P_mean is None and P_var is None and P_min is None and P_max is None and P_negative_count is None and P_negative_sum is None) or (type(P_mean) is float and type(P_var) is float and type(P_min) is float and type(P_max) is float and ((self._deviationNegativeValues and type(P_negative_count) is int and type(P_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert (Z_mean is None and Z_var is None and Z_min is None and Z_max is None and Z_negative_count is None and Z_negative_sum is None) or (type(Z_mean) is float and type(Z_var) is float and type(Z_min) is float and type(Z_max) is float and ((self._deviationNegativeValues and type(Z_negative_count) is int and type(Z_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert (D_mean is None and D_var is None and D_min is None and D_max is None and D_negative_count is None and D_negative_sum is None) or (type(D_mean) is float and type(D_var) is float and type(D_min) is float and type(D_max) is float and ((self._deviationNegativeValues and type(D_negative_count) is int and type(D_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year FROM DeviationTracer WHERE simulationId = ? AND year = ?'
        #Test, if dataset for this simulationId, year combination exists
        if overwrite:
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            #Remove database entry for this simulationId, year combination
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM DeviationTracer WHERE simulationId = ? AND year = ?'
                self._c.execute(sqlcommand, (simulationId, year))
        else:
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate insert
        purchases = []
        if self._deviationNegativeValues:
            purchases.append((simulationId, year, N_mean, N_var, N_min, N_max, N_negative_count, N_negative_sum, DOP_mean, DOP_var, DOP_min, DOP_max, DOP_negative_count, DOP_negative_sum, P_mean, P_var, P_min, P_max, P_negative_count, P_negative_sum, Z_mean, Z_var, Z_min, Z_max, Z_negative_count, Z_negative_sum, D_mean, D_var, D_min, D_max, D_negative_count, D_negative_sum))
        else:
            purchases.append((simulationId, year, N_mean, N_var, N_min, N_max, DOP_mean, DOP_var, DOP_min, DOP_max, P_mean, P_var, P_min, P_max, Z_mean, Z_var, Z_min, Z_max, D_mean, D_var, D_min, D_max))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO DeviationTracer VALUES {:s}'.format('(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)' if self._deviationNegativeValues else '(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def _create_table_DeviationTracerDifference(self):
        """
        Create table DeviationTracerDifference

        Notes
        -----
        If attribute self._deviationNegativeValues is True, create two columns
        for the number of boxes with negative concentrations and the sum of
        the negative concentrations for each tracer.
        """
        if self._deviationNegativeValues:
            self._c.execute('''CREATE TABLE DeviationTracerDifference (simulationIdA INTEGER NOT NULL REFERENCES Simulation(simulationId), simulationIdB INTEGER NOT NULL REFERENCES Simulation(simulationId), yearA INTEGER NOT NULL, yearB INTEGER NOT NULL, N_mean REAL NOT NULL, N_var REAL NOT NULL, N_min REAL NOT NULL, N_max REAL NOT NULL, N_negative_count INTEGER NOT NULL, N_negative_sum REAL NOT NULL, DOP_mean REAL, DOP_var REAL, DOP_min REAL, DOP_max REAL, DOP_negative_count INTEGER, DOP_negative_sum REAL, P_mean REAL, P_var REAL, P_min REAL, P_max REAL, P_negative_count INTEGER, P_negative_sum REAL, Z_mean REAL, Z_var REAL, Z_min REAL, Z_max REAL, Z_negative_count INTEGER, Z_negative_sum REAL, D_mean REAL, D_var REAL, D_min REAL, D_max REAL, D_negative_count INTEGER, D_negative_sum REAL, PRIMARY KEY (simulationIdA, simulationIdB, yearA, yearB))''')
        else:
            self._c.execute('''CREATE TABLE DeviationTracerDifference (simulationIdA INTEGER NOT NULL REFERENCES Simulation(simulationId), simulationIdB INTEGER NOT NULL REFERENCES Simulation(simulationId), yearA INTEGER NOT NULL, yearB INTEGER NOT NULL, N_mean REAL NOT NULL, N_var REAL NOT NULL, N_min REAL NOT NULL, N_max REAL NOT NULL, DOP_mean REAL, DOP_var REAL, DOP_min REAL, DOP_max REAL, P_mean REAL, P_var REAL, P_min REAL, P_max REAL, Z_mean REAL, Z_var REAL, Z_min REAL, Z_max REAL, D_mean REAL, D_var REAL, D_min REAL, D_max REAL, PRIMARY KEY (simulationIdA, simulationIdB, yearA, yearB))''')


    def check_difference_tracer_deviation(self, simulationIdA, simulationIdB, expectedCount):
        """
        Check number of tracer difference deviation entries for a simulationId

        Parameters
        ----------
        simulationIdA : int
            Id defining the parameter for the first spin up calculation
        simulationIdB : int
            Id defining the parameter for the second spin up calculation
        expectedCount : int
            Expected number of database entries for the spin up

        Returns
        -------
        bool
           True if number of database entries coincides with the expected
           number
        """
        assert type(simulationIdA) is int and simulationIdA >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM DeviationTracerDifference WHERE simulationIdA = ? AND simulationIdB = ?'
        self._c.execute(sqlcommand, (simulationIdA, simulationIdB))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            print('CheckTracerDifferenceDeviation: Expected: {} Get: {}'.format(expectedCount, count[0][0]))

        return count[0][0] == expectedCount


    def insert_difference_tracer_deviation_tuple(self, simulationIdA, simulationIdB, yearA, yearB, N_mean, N_var, N_min, N_max, N_negative_count=None, N_negative_sum=None, DOP_mean=None, DOP_var=None, DOP_min=None, DOP_max=None, DOP_negative_count=None, DOP_negative_sum=None, P_mean=None, P_var=None, P_min=None, P_max=None, P_negative_count=None, P_negative_sum=None, Z_mean=None, Z_var=None, Z_min=None, Z_max=None, Z_negative_count=None, Z_negative_sum=None, D_mean=None, D_var=None, D_min=None, D_max=None, D_negative_count=None, D_negative_sum=None, overwrite=False):
        """
        Insert deviation for a tracer difference

        Insert deviation for a tracer difference. If a database entry for the
        deviation between the tracers of the spin simulations with the
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
        N_mean : float
            Mean of the N tracer
        N_var : float
            Variance of the N tracer
        N_min : float
            Minimum of the N tracer
        N_max : float
            Maximum of the N tracer
        N_negative_count : int or None, default: None
            Number of boxes for which the concentration of the N tracer is
            negative
        N_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the N tracer
        DOP_mean : float or None, default: None
            Mean of the DOP tracer
        DOP_var : float or None, default: None
            Variance of the DOP tracer
        DOP_min : float or None, default: None
            Minimum of the DOP tracer
        DOP_max : float or None, default: None
            Maximum of the DOP tracer
        DOP_negative_count : int or None, default: None
            Number of boxes for which the concentration of the DOP tracer is
            negative
        DOP_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the DOP tracer
        P_mean : float or None, default: None
            Mean of the P tracer
        P_var : float or None, default: None
            Variance of the P tracer
        P_min : float or None, default: None
            Minimum of the P tracer
        P_max : float or None, default: None
            Maximum of the P tracer
        P_negative_count : int or None, default: None
            Number of boxes for which the concentration of the P tracer is
            negative
        P_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the P tracer
        Z_mean : float or None, default: None
            Mean of the Z tracer
        Z_var : float or None, default: None
            Variance of the Z tracer
        Z_min : float or None, default: None
            Minimum of the Z tracer
        Z_max : float or None, default: None
            Maximum of the Z tracer
        Z_negative_count : int or None, default: None
            Number of boxes for which the concentration of the Z tracer is
            negative
        Z_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the Z tracer
        D_mean : float or None, default: None
            Mean of the D tracer
        D_var : float or None, default: None
            Variance of the D tracer
        D_min : float or None, default: None
            Minimum of the D tracer
        D_max : float or None, default: None
            Maximum of the D tracer
        D_negative_count : int or None, default: None
            Number of boxes for which the concentration of the D tracer is
            negative
        D_negative_sum : float or None, default: None
            Sum of all negative tracer concentrations of the D tracer
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
        assert type(N_mean) is float and type(N_var) is float and type(N_min) is float and type(N_max) is float and ((self._deviationNegativeValues and type(N_negative_count) is int and type(N_negative_sum) is float) or (not self._deviationNegativeValues))
        assert (DOP_mean is None and DOP_var is None and DOP_min is None and DOP_max is None and DOP_negative_count is None and DOP_negative_sum is None) or (type(DOP_mean) is float and type(DOP_var) is float and type(DOP_min) is float and type(DOP_max) is float and ((self._deviationNegativeValues and type(DOP_negative_count) is int and type(DOP_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert (P_mean is None and P_var is None and P_min is None and P_max is None and P_negative_count is None and P_negative_sum is None) or (type(P_mean) is float and type(P_var) is float and type(P_min) is float and type(P_max) is float and ((self._deviationNegativeValues and type(P_negative_count) is int and type(P_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert (Z_mean is None and Z_var is None and Z_min is None and Z_max is None and Z_negative_count is None and Z_negative_sum is None) or (type(Z_mean) is float and type(Z_var) is float and type(Z_min) is float and type(Z_max) is float and ((self._deviationNegativeValues and type(Z_negative_count) is int and type(Z_negative_sum) is float) or (not self._deviationNegativeValues)))
        assert (D_mean is None and D_var is None and D_min is None and D_max is None and D_negative_count is None and D_negative_sum is None) or (type(D_mean) is float and type(D_var) is float and type(D_min) is float and type(D_max) is float and ((self._deviationNegativeValues and type(D_negative_count) is int and type(D_negative_sum) is float) or (not self._deviationNegativeValues)))

        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationIdA, simulationIdB, yearA, yearB FROM DeviationTracerDifference WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?'
        #Test, if dataset for this simulationIdA, simulationIdB, yearA, yearB combination exists
        if overwrite:
            self._c.execute(sqlcommand_select, (simulationIdA, simulationIdB, yearA, yearB))
            dataset = self._c.fetchall()
            #Remove database entry for this simulationIdA, simulationIdB, yearA, yearB combination
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM DeviationTracerDifference WHERE simulationIdA = ? AND simulationIdB = ? AND yearA = ? AND yearB = ?'
                self._c.execute(sqlcommand, (simulationIdA, simulationIdB, yearA, yearB))
        else:
            self._c.execute(sqlcommand_select, (simulationIdA, simulationIdB, yearA, yearB))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate insert
        purchases = []
        if self._deviationNegativeValues:
            purchases.append((simulationIdA, simulationIdB, yearA, yearB, N_mean, N_var, N_min, N_max, N_negative_count, N_negative_sum, DOP_mean, DOP_var, DOP_min, DOP_max, DOP_negative_count, DOP_negative_sum, P_mean, P_var, P_min, P_max, P_negative_count, P_negative_sum, Z_mean, Z_var, Z_min, Z_max, Z_negative_count, Z_negative_sum, D_mean, D_var, D_min, D_max, D_negative_count, D_negative_sum))
        else:
            purchases.append((simulationIdA, simulationIdB, yearA, yearB, N_mean, N_var, N_min, N_max, DOP_mean, DOP_var, DOP_min, DOP_max, P_mean, P_var, P_min, P_max, Z_mean, Z_var, Z_min, Z_max, D_mean, D_var, D_min, D_max))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO DeviationTracerDifference VALUES {:s}'.format('(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)' if self._deviationNegativeValues else '(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'), purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)

