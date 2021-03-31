#!/usr/bin/env python
# -*- coding: utf8 -*

import copy
import os
import numpy as np

import initialValue.constants as InitialValue_Constants
import metos3dutil.database.constants as DB_Constants
from metos3dutil.database.DatabaseMetos3d import DatabaseMetos3d
import metos3dutil.metos3d.constants as Metos3d_Constants


class InitialValue_Database(DatabaseMetos3d):
    """
    Access functions for the database
    """

    def __init__(self, dbpath=InitialValue_Constants.DB_PATH, completeTable=True, createDb=False):
        """
        Initialization of the database connection

        Parameter
        ----------
        dbpath : str, default: initalValue.constants.DB_PATH
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
        Create all table of the database
        """
        self._c.execute('PRAGMA foreign_keys=on')
        self._create_table_parameter()
        self._create_table_initialConcentration()
        self._create_table_simulation()
        self._create_table_spinup()

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


    def _init_tables_initialConcentration_Simulation(self):
        """
        Initial insert into table InitialConcentration and Simulation
        """
        initialConcentration = []
        concentrationId = 0
        concentrationNum = {}
        concentrationNum['constant'] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        concentrationNum['vector'] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        simulations = []
        simulationId = 0
        parameterId = 0
        timestep = 1

        #Standard constant initial concentrations
        concentrationTyp = 'constant'
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS[:-1]:
            concentration = [Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][0]] + Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][1:-1]
            while len(concentration) < len(Metos3d_Constants.TRACER_MASK)-1:
                concentration.append(None)
            concentration.append(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel][-1] if len(Metos3d_Constants.INITIAL_CONCENTRATION[metos3dModel]) > 1 else None)

            initialConcentration.append((concentrationId, concentrationTyp, len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]), concentrationNum[concentrationTyp][len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel])], None, None) + tuple(concentration))
            simulations.append((simulationId, metos3dModel, parameterId, concentrationId, timestep))
            concentrationId += 1
            concentrationNum[concentrationTyp][len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel])] += 1
            simulationId += 1

        #Simulation for the MITgcm-PO4-DOP model using the standard constant initial concentrations
        simulations.append((simulationId, Metos3d_Constants.METOS3D_MODELS[-1], parameterId, 1, timestep))
        simulationId += 1

        #Initial tracer concentration vector
        tracerModelDict = {1: 'N', 2: 'N-DOP', 3: 'NP-DOP', 4: 'NPZ-DOP', 5: 'NPZD-DOP'}
        concentrationTyp = 'vector'
        for tracerDistribution in Metos3d_Constants.METOS3D_MODEL_TRACER_TRACERDISTRIBUTION:
            for distribution in Metos3d_Constants.METOS3D_MODEL_TRACER_DISTRIBUTION:
                simulationsMITgcm = []
                for differentTracer in range(1, len(Metos3d_Constants.TRACER_MASK)+1):
                    for tracerNum in range(0, 100):
                        concentration = [concentrationId, concentrationTyp, differentTracer, concentrationNum[concentrationTyp][differentTracer], distribution, tracerDistribution, None, None, None, None, None]

                        #Generate row of initialConcentration table
                        for differentTracerNum in range(0, differentTracer):
                            if differentTracer > 1 and differentTracerNum + 1 == differentTracer:
                                concentration[-1] = InitialValue_Constants.PATTERN_TRACER_INITIAL_CONCENTRATION.format(differentTracerNum, tracerNum)
                            else:
                                concentration[6 + differentTracerNum] = InitialValue_Constants.PATTERN_TRACER_INITIAL_CONCENTRATION.format(differentTracerNum, tracerNum)
                        initialConcentration.append(tuple(concentration))

                        #Generate rows of simulation table
                        simulations.append((simulationId, tracerModelDict[differentTracer], parameterId, concentrationId, timestep))

                        #Generate rows for the MITgcm-PO4-DOP model
                        if tracerModelDict[differentTracer] == 'N-DOP':
                            simulationsMITgcm.append((Metos3d_Constants.METOS3D_MODELS[-1], parameterId, concentrationId, timestep))

                        concentrationId += 1
                        concentrationNum[concentrationTyp][differentTracer] += 1
                        simulationId += 1

                #Rows for the MITgcm-PO4-DOP model
                for t in simulationsMITgcm:
                    simulations.append((simulationId,) + t)
                    simulationId += 1

        #Random constant initial concentration
        #TODO Analoges Vorgehen zum LatinHypercubeSample implementieren (Modul zum Erzeugen von Anfangskonzentrationen und diese in eine Datei schreiben/auslesen) Hier fuer die Initialbefuellung auslesen

        #Systematic adjustment of the constant initial concentration
        concentrationTyp = 'constant'
        simulationsMITgcm = []
        for differentTracer in range(2, len(Metos3d_Constants.TRACER_MASK)+1):
            for i in range(differentTracer):
                concentration = copy.deepcopy(Metos3d_Constants.INITIAL_CONCENTRATION[tracerModelDict[differentTracer]])
                concentration[0] = Metos3d_Constants.INITIAL_CONCENTRATION[tracerModelDict[differentTracer]][1]
                concentration[i] = Metos3d_Constants.INITIAL_CONCENTRATION[tracerModelDict[differentTracer]][0]

                stepSize = 0.1
                while concentration[i] > 0.0:
                    concentrationTupel = [concentration[0]] + concentration[1:-1]
                    while len(concentrationTupel) < len(Metos3d_Constants.TRACER_MASK)-1:
                        concentrationTupel.append(None)
                    concentrationTupel.append(concentration[-1])

                    #Do not insert the standard constant concentration again
                    if not concentrationTupel[0] == Metos3d_Constants.INITIAL_CONCENTRATION[tracerModelDict[differentTracer]][0]:
                        initialConcentration.append((concentrationId, concentrationTyp, differentTracer, concentrationNum[concentrationTyp][differentTracer], None, None) + tuple(concentrationTupel))
                        simulations.append((simulationId, tracerModelDict[differentTracer], parameterId, concentrationId, timestep))
                        simulationId += 1

                        #Simulation for the MITgcm-PO4-DOP
                        if tracerModelDict[differentTracer] == 'N-DOP':
                            simulationsMITgcm.append((Metos3d_Constants.METOS3D_MODELS[-1], parameterId, concentrationId, timestep))

                        concentrationId += 1
                        concentrationNum[concentrationTyp][differentTracer] += 1

                    #Add stepSize to the component i in subtract it in the next step
                    concentration = [l + stepSize for l in concentration]
                    concentration[i] -= stepSize * differentTracer

        for t in simulationsMITgcm:
            simulations.append((simulationId,) + t)
            simulationId += 1

        self._c.executemany('INSERT INTO InitialConcentration VALUES (?,?,?,?,?,?,?,?,?,?,?)', initialConcentration)
        self._c.executemany('INSERT INTO Simulation VALUES (?,?,?,?,?)', simulations)

        self._conn.commit()


    def _init_database(self):
        """
        Initial insert of the database tables

        Notes
        -----
        The functions inserts data sets into the tables Parameter,
        InitialConcentration and Simulation
        """
        #Insert the reference parameter set and the parameter of the latin hypercube sample with 100 samples into the table Parameter
        self._init_table_parameter(referenceParameter=True, latinHypercubeSamples=(True, False, False))

        #Insert initial concentration and simulation data sets
        self._init_tables_initialConcentration_Simulation()


    def _create_table_initialConcentration(self):
        """
        Create table InitialConcentration
        """
        self._c.execute('''CREATE TABLE InitialConcentration (concentrationId INTEGER NOT NULL, concentrationTyp TEXT NOT NULL, differentTracer INTEGER NOT NULL, concentrationNum INTEGER NOT NULL, distribution TEXT, tracerDistribution TEXT, N TEXT NOT NULL, P TEXT, Z TEXT, D TEXT, DOP TEXT, PRIMARY KEY (concentrationId), UNIQUE (concentrationTyp, differentTracer, concentrationNum))''')


    def exists_initialConcentration(self, concentrationTyp, differentTracer, concentrationNum):
        """
        Returns if a database entry exists for the initial concentration

        Parameters
        ----------
        concentrationTyp : {'vector', 'constant'}
            Use constant initial concentration or an initial concentration
            defined with vectors
        differentTracer : int
            Number of different tracer
        concentrationNum : int
            Index of the concentration

        Returns
        -------
        bool
            True if an entry exists for the given concentration
        """
        assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
        assert type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.TRACER_MASK)
        assert type(concentrationNum) is int and 0 <= concentrationNum

        sqlcommand = 'SELECT concentrationId FROM InitialConcentration WHERE concentrationTyp = ? AND differentTracer = ? AND concentrationNum = ?'
        self._c.execute(sqlcommand, (concentrationTyp, differentTracer, concentrationNum))
        concentrationId = self._c.fetchall()
        return len(concentrationId) > 0


    def exists_initialConcentration(self, concentrationTyp, differentTracer, N, distribution=None, tracerDistribution=None, P=None, Z=None, D=None, DOP=None):
        """
        Returns if a database entry exists for the given concentration

        Parameters
        ----------
        concentrationTyp : {'vector', 'constant'}
            Use constant initial concentration or an initial concentration
            defined with vectors
        differentTracer : int
            Number of the different tracer
        N : str or float
            If concentrationTyp is 'constant', global mean concentration of
            the N tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the N tracer.
        distribution : {None, 'Uniform', 'Normal', 'Lognormal', 'OneBox'},
            default: None
            If concentrationTyp is 'vector', distribution used to generate the
            initial tracer concentration vectors.
        tracerDistribution : {None, 'set_mass', 'random_mass'}, default: None
            TODO
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
        assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
        assert type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.TRACER_MASK)
        assert distribution is None or distribution in Metos3d_Constants.METOS3D_MODEL_TRACER_DISTRIBUTION
        assert tracerDistribution is None or tracerDistribution in Metos3d_Constants.METOS3D_MODEL_TRACER_TRACERDISTRIBUTION
        assert (concentrationTyp == 'constant' and distribution is None and tracerDistribution is None) or (concentrationTyp == 'vector' and distribution is not None and tracerDistribution is not None)
        assert concentrationTyp == 'constant' and type(N) is float or concentrationTyp == 'vector' and type(N) is str
        assert P is None or concentrationTyp == 'constant' and type(P) is float or concentrationTyp == 'vector' and type(P) is str
        assert Z is None or concentrationTyp == 'constant' and type(Z) is float or concentrationTyp == 'vector' and type(Z) is str
        assert D is None or concentrationTyp == 'constant' and type(D) is float or concentrationTyp == 'vector' and type(D) is str
        assert DOP is None or concentrationTyp == 'constant' and type(DOP) is float or concentrationTyp == 'vector' and type(DOP) is str

        sqlcommand = 'SELECT concentrationId FROM InitialConcentration WHERE concentrationTyp = ? AND differentTracer = ?'
        sqltuple = (concentrationTyp, differentTracer)

        if distribution is None:
            sqlcommand = sqlcommand + ' AND distribution IS NULL'
        else:
            sqlcommand = sqlcommand + ' AND distribution = ?'
            sqltuple = sqltupel + (distribution,)

        if tracerDistribution is None:
            sqlcommand = sqlcommand + ' AND tracerDistribution IS NULL'
        else:
            sqlcommand = sqlcommand + ' AND tracerDistribution = ?'
            sqltuple = sqltupel + (tracerDistribution,)

        sqlcommand = sqlcommand + ' AND N = ?'
        sqltupel = sqltupel + (N,)

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


    def get_concentrationId(self, concentrationTyp, differentTracer, concentrationNum):
        """
        Returns the concentrationId of the initial concentration

        Parameters
        ----------
        concentrationTyp : {'vector', 'constant'}
            Use constant initial concentration or an initial concentration
            defined with vectors
        differentTracer : int
            Number of different tracer
        concentrationNum : int
            Index of the concentration

        Returns
        -------
        int
            concentrationId of the initial tracer concentration

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
        assert type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.TRACER_MASK)
        assert type(concentrationNum) is int and 0 <= concentrationNum

        sqlcommand = 'SELECT concentrationId FROM InitialConcentration WHERE concentrationTyp = ? AND differentTracer = ? AND concentrationNum = ?'
        self._c.execute(sqlcommand, (concentrationTyp, differentTracer, concentrationNum))
        concentrationId = self._c.fetchall()
        assert len(concentrationId) == 1
        return concentrationId[0][0]


    def get_distribution(self, concentrationId):
        """
        Returns the distribution and tracerDistribution

        Parameters
        ----------
        concentrationId : int
            Id of the initial tracer concentration

        Returns
        -------
        tuple [str]
            Tuple consisting of the distribution and tracerDistribution

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
        """
        assert type(concentrationId) is int and 0 <= concentrationId

        sqlcommand = 'SELECT distribution, tracerDistribution FROM InitialConcentration WHERE concentrationId = ?'
        self._c.execute(sqlcommand, (concentrationId, ))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (dataset[0][0], dataset[0][1])


    def insert_initialConcentration(self, concentrationTyp, differentTracer, N, distribution=None, tracerDistribution=None, P=None, Z=None, D=None, DOP=None):
        """
        Insert initial concentration values

        Parameters
        ----------
        concentrationTyp : {'vector', 'constant'}
            Use constant initial concentration or an initial concentration
            defined with vectors
        differentTracer : int
            Number of different tracer
        N : str or float
            If concentrationTyp is 'constant', global mean concentration of
            the N tracer in each box of the ocean discretization. Otherweise
            the name of the initial concentration vector for the N tracer.
        distribution : {None, 'Uniform', 'Normal', 'Lognormal', 'OneBox'},
            default: None
            If concentrationTyp is 'vector', distribution used to generate the
            initial tracer concentration vectors.
        tracerDistribution : {None, 'set_mass', 'random_mass'}, default: None
            TODO
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
        assert type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.TRACER_MASK)
        assert distribution is None or distribution in Metos3d_Constants.METOS3D_MODEL_TRACER_DISTRIBUTION
        assert tracerDistribution is None or tracerDistribution in Metos3d_Constants.METOS3D_MODEL_TRACER_TRACERDISTRIBUTION
        assert (concentrationTyp == 'constant' and distribution is None and tracerDistribution is None) or (concentrationTyp == 'vector' and distribution is not None and tracerDistribution is not None)
        assert concentrationTyp == 'constant' and type(N) is float or concentrationTyp == 'vector' and type(N) is str
        assert P is None or concentrationTyp == 'constant' and type(P) is float or concentrationTyp == 'vector' and type(P) is str
        assert Z is None or concentrationTyp == 'constant' and type(Z) is float or concentrationTyp == 'vector' and type(Z) is str
        assert D is None or concentrationTyp == 'constant' and type(D) is float or concentrationTyp == 'vector' and type(D) is str
        assert DOP is None or concentrationTyp == 'constant' and type(DOP) is float or concentrationTyp == 'vector' and type(DOP) is str

        if not self.exists_initialConcentration(concentrationTyp, differentTracer, N, distribution=distribution, tracerDistribution=tracerDistribution, P=P, Z=Z, D=D, DOP=DOP):
            #Insert initial concentration into the database
            #Determine concentrationId
            sqlcommand = 'SELECT MAX(concentrationId) FROM InitialConcentration'
            self._c.execute(sqlcommand)
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            concentrationId = dataset[0][0] + 1

            #Determine concentrationNum
            sqlcommand = 'SELECT MAX(concentrationNum) FROM InitialConcentration WHERE concentrationTyp = ? AND differentTracer = ?'
            self._c.execute(sqlcommand, (concentrationTyp, differentTracer))
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            concentrationNum = dataset[0][0] + 1

            purchases = [(concentrationId, concentrationTyp, differentTracer, concentrationNum, distribution, tracerDistribution, N, P, Z, D, DOP)]

            inserted = False
            insertCount = 0
            while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
                try:
                    self._c.executemany('INSERT INTO InitialConcentration VALUES (?,?,?,?,?,?,?,?,?,?,?)', purchases)
                    self._conn.commit()
                    inserted = True
                except sqlite3.OperationalError:
                    insertCount += 1
                    #Wait for the next insert
                    time.sleep(DB_Constants.TIME_SLEEP)


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

