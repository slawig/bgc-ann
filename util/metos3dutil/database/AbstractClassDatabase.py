#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import logging
import numpy as np
import os
import sqlite3

import metos3dutil.database.constants as DB_Constants
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


class AbstractClassDatabase(ABC):
    """
    Abstract class for the database access.
    @author: Markus Pfeil
    """

    def __init__(self, dbpath, completeTable=True):
        """
        Initialization of the database connection
        @author: Markus Pfeil
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        self._conn = sqlite3.connect(dbpath, timeout=DB_Constants.timeout)
        self._c = self._conn.cursor()

        self._completeTable = completeTable


    def close_connection(self):
        """
        Close the database connection
        @author: Markus Pfeil
        """
        self._conn.close()


    def get_parameterId(self, parameter, metos3dModel):
        """
        Get parameterId for given model and parameter values.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list

        modelParameter = Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        assert len(modelParameter) == len(parameter)

        sqlcommand = 'SELECT parameterId FROM Parameter WHERE ' + modelParameter[0] + ' = ?'
        for i in range(1, len(parameter)):
            sqlcommand = sqlcommand + ' AND ' + modelParameter[i] + ' = ?'
        self._c.execute(sqlcommand, tuple(parameter))
        parameterId = self._c.fetchall()
        assert len(parameterId) == 1
        return parameterId[0][0]


    def get_parameter(self, parameterId, metos3dModel):
        """
        Get parameter values of the given model for a given parameterId.
        @author: Markus Pfeil
        """
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        sqlcommand = 'SELECT * FROM Parameter WHERE parameterId = ?'
        self._c.execute(sqlcommand, (parameterId, ))
        para = self._c.fetchall()
        assert len(para) == 1
        modelParameterAll = para[0][1:]
        modelParameter = np.array(modelParameterAll)[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        parameter = []
        for i in range(len(modelParameter)):
            if modelParameter[i] is not None:
                parameter.append(modelParameter[i])

        if metos3dModel == 'MITgcm-PO4-DOP':
            assert len(parameter) == 7
            parameter = [parameter[5], parameter[1], parameter[4], parameter[2], parameter[3], parameter[0], parameter[6]]

        assert len(parameter) == len(Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]])
        return np.array(parameter)


    def insert_parameter(self, parameter, metos3dModel):
        """
        Insert parameter values for the given model
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list and len(parameter) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[metos3dModel]

        sqlcommand = 'SELECT MAX(parameterId) FROM Parameter'
        self._c.execute(sqlcommand)
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        parameterId = dataset[0][0] + 1

        if metos3dModel == 'MITgcm-PO4-DOP':
            parameter = [parameter[5], parameter[1], parameter[4], parameter[2], parameter[3], parameter[0], parameter[6]]

        purchases = []
        modelParameter = [None for _ in range(20)]
        i = 0
        for j in range(20):
            if Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel][j]:
                modelParameter[j] = parameter[i]
                i = i + 1
        assert i == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[metos3dModel]

        purchases.append((parameterId,) + tuple(modelParameter))
        self._c.executemany('INSERT INTO Parameter VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
        self._conn.commit()

        return parameterId


    def get_concentrationId_constantValues(self, metos3dModel, concentrationValues):
        """
        Get concentrationId for constant initial values.
        @author: Markus Pfeil
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
        Get concentrationId for vector intial values.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert 0 <= tracerNum and tracerNum < 100
        assert distribution in ['Lognormal', 'Normal', 'OneBox', 'Uniform']
        assert tracerDistribution in ['set_mass', 'random_mass']

        concentrationParameter = Metos3d_Constants.TRACER_MASK[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[metos3dModel]]
        sqlcommand = 'SELECT concentrationId FROM InitialConcentration WHERE concentration_typ = ? AND distribution = ? AND tracerDistribution = ? AND differentTracer = ?'
        for i in range(0, len(concentrationParameter)):
            sqlcommand = sqlcommand + " AND {} = 'InitialValue_Tracer_{}_{:0>3}.petsc'".format(concentrationParameter[i], i, tracerNum)
        self._c.execute(sqlcommand, ('vector', distribution, tracerDistribution, self.DifferentTracer[metos3dModel])) #TODO Reset DifferentTracer
        concentrationId = self._c.fetchall()
        assert len(concentrationId) == 1
        return concentrationId[0][0]


    @abstractmethod
    def get_simulationId(self, model, parameterId, concentrationId):
        """
        Get the simulationId for the given values.
        @author: Markus Pfeil
        """
        pass


    def get_spinup_year_for_tolerance(self, simulationId, tolerance=10**(-4)):
        """
        Get the year of the spin up where the spin up tolerance fall below the given tolerance.
        If the tolerance of the spin up is higher than the given tolerance for every model year, return None.
        @author: Markus Pfeil
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


    def check_spinup(self, simulationId, expectedCount):
        """
        Check the number of entries in the table spin-up with the given simulationId.
        @author: Markus Pfeil
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
        Insert spin-up value
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(tolerance) is float and 0 <= tolerance
        assert type(spinupNorm) is float and 0 <= spinupNorm
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT simulationId, year FROM Spinup WHERE simulationId = ? AND year = ?'
        if overwrite:
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Spinup WHERE simulationId = ? AND year = ?'
                self._c.execute(sqlcommand, (simulationId, year))
        else:
            #Test, if dataset for this simulationId and year combination exists
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            assert len(dataset) == 0
        #Generate and insert spin-up value
        purchases = []
        purchases.append((simulationId, year, tolerance, spinupNorm))
        self._c.executemany('INSERT INTO Spinup VALUES (?,?,?,?)', purchases)
        self._conn.commit()


    def check_tracer_norm(self, simulationId, expectedCount, norm='2', trajectory=''):
        """
        Check the number of entries in the table of the tracer with the given norm for the given simulationId
        @author: Markus Pfeil
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
        Insert tracer norm values.
        @author: Markus Pfeil
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
        if overwrite:
            #Test, if dataset for this simulationId and year combination exists
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            #Remove database entry for this simulationId, year
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Tracer{}{}Norm WHERE simulationId = ? AND year = ?'.format(trajectory, norm)
                self._c.execute(sqlcommand, (simulationId, year))
        else:
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate insert for the tracer norm
        purchases = []
        purchases.append((simulationId, year, tracer, N, DOP, P, Z, D))
        self._c.executemany('INSERT INTO Tracer{}{}Norm VALUES (?,?,?,?,?,?,?,?)'.format(trajectory, norm), purchases)
        self._conn.commit()


    def check_difference_tracer_norm(self, simulationIdA, simulationIdB, expectedCount, norm='2', trajectory=''):
        """
        Check the number of entries in the table of the tracer differences with the given norm for the given simulationIdA and simulationIdB
        @author: Markus Pfeil
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
        Insert the norm of a difference of two tracers
        @author: Markus Pfeil
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
        self._c.executemany('INSERT INTO TracerDifference{}{}Norm VALUES (?,?,?,?,?,?,?,?,?,?)'.format(trajectory, norm), purchases)
        self._conn.commit()


    def read_spinup_values_for_simid(self, simulationId):
        """
        Read values of the spin up for the given simulationId.
        @author: Markus Pfeil
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
    def read_rel_norm(self, model, concentrationId, year=None, norm='2', parameterId=None, trajectory=''):
        """
        Read for every parameterId the norm value for the given annId from the database.
        If parameterId is not None, read only the relative difference for the given parameterId.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def read_spinup_tolerance(self, model, concentrationId, year):
        """
        Read the spin up tolerance for every parameterId of the given year for the given ann setting.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def read_spinup_year(self, model, concentrationId):
        """
        Read the required years to reach the given spin up tolerance for every parameterId for the given ann setting.
        @author: Markus Pfeil
        """
        pass
