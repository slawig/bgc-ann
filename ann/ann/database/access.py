#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
import numpy as np
import os
import sqlite3
import time

from metos3dutil.database.DatabaseMetos3d import DatabaseMetos3d
import ann.network.constants as ANN_Constants
import ann.database.constants as ANN_DB_Constants
import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


class Ann_Database(DatabaseMetos3d):
    """
    Class for the database access.
    @author: Markus Pfeil
    """

    def __init__(self, dbpath=ANN_DB_Constants.dbPath, completeTable=True):
        """
        Initialization of the database connection
        @author: Markus Pfeil
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        DatabaseMetos3d.__init__(self, dbpath, completeTable=completeTable)


    def get_concentrationId_annValues(self, annId):
        """
        Get concentrationId for the predicition of the ann as initial values.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        sqlcommand = "SELECT concentrationId FROM InitialConcentration WHERE concentration_typ = 'ann' AND annId = ?"
        self._c.execute(sqlcommand, (annId, ))
        concentrationId = self._c.fetchall()
        assert len(concentrationId) == 1
        return concentrationId[0][0]


    def get_annTypeModel(self, annId):
        """
        Get the annType and the model for a given annId.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        sqlcommand = 'SELECT annType, model FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1])


    def get_annTypeNumberModel(self, annId):
        """
        Get the annType, the annNumber and the model for a given annId.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        sqlcommand = 'SELECT annType, annNumber, model FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1], annConfig[0][2])


    def get_annTypeNumberModelMass(self, annId):
        """
        Get the annType, the annNumber, the model and the conserve mass flag for a given annId.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        sqlcommand = 'SELECT annType, annNumber, model, conserveMass FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1], annConfig[0][2], bool(annConfig[0][3]))


    def get_annConfig(self, annId):
        """
        Get the parameter of the ann configuration for the given annId.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        sqlcommand = 'SELECT annType, annNumber, model, validationSplit, kfoldSplit, epochs, trainingData FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1], annConfig[0][2], annConfig[0][3], annConfig[0][4], annConfig[0][5], annConfig[0][6])


    def get_simulationId(self, model, parameterId, concentrationId, massCorrection=False, tolerance=None, cpus=64):
        """
        Get the simulationId for the given values.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(concentrationId) is int and concentrationId >= 0
        assert type(massCorrection) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0

        if tolerance is None:
            sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND massCorrection = ? AND tolerance is NULL AND cpus = ?'
            self._c.execute(sqlcommand, (model, parameterId, concentrationId, int(massCorrection), cpus))
        else:
            sqlcommand = 'SELECT simulationId FROM Simulation WHERE model = ? AND parameterId = ? AND concentrationId = ? AND massCorrection = ? AND tolerance = ? AND cpus = ?'
            self._c.execute(sqlcommand, (model, parameterId, concentrationId, int(massCorrection), tolerance, cpus))
        simulationId = self._c.fetchall()

        assert len(simulationId) == 1
        return simulationId[0][0]


    def check_AnnTraining(self, annId, expectedCount=None):
        """
        Check the number of entries in the table AnnTraining with the given annId.
        @author: Markus Pfeil
        """
        assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert expectedCount is None or type(expectedCount) is int and 0 <= expectedCount

        sqlcommand = 'SELECT COUNT(*) FROM AnnTraining WHERE annId = ?'
        self._c.execute(sqlcommand, (annId, ))
        count = self._c.fetchall()
        assert len(count) == 1

        if expectedCount is None:
            ret = count[0][0] > 0
        else:
            if count[0][0] != expectedCount:
                logging.info('***CheckAnnTraining: Expected: {} Get: {}***'.format(expectedCount, count[0][0]))
            ret = count[0][0] == expectedCount

        return ret


    def insert_training(self, annId, epoch, loss, valLoss, meanSquaredError, valMeanSquaredError, meanAbsoulteError, valMeanAbsoluteError, overwrite=False):
        """
        Insert values of the loss functions for a epoch of the training.
        @author: Markus Pfeil
        """
        assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(epoch) is int and 0 <= epoch
        assert type(loss) is float and 0.0 <= loss
        assert type(valLoss) is float and 0.0 <= valLoss
        assert type(meanSquaredError) is float and 0.0 <= meanSquaredError
        assert type(valMeanSquaredError) is float and 0.0 <= valMeanSquaredError
        assert type(meanAbsoulteError) is float and 0.0 <= meanAbsoulteError
        assert type(valMeanAbsoluteError) is float and 0.0 <= valMeanAbsoluteError
        assert type(overwrite) is bool

        sqlcommand_select = 'SELECT annId, epoch FROM AnnTraining WHERE annId = ? AND epoch = ?'
        if overwrite:
            self._c.execute(sqlcommand_select, (annId, epoch))
            dataset = self._c.fetchall()
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM AnnTraining WHERE annId = ? AND epoch = ?'
                self._c.execute(sqlcommand, (annId, epoch))
        else:
            #Test, if dataset for this annId and epoch combination exists
            self._c.execute(sqlcommand_select, (annId, epoch))
            dataset = self._c.fetchall()
            assert len(dataset) == 0

        #Generate and insert training value
        purchases = []
        purchases.append((annId, epoch, loss, valLoss, meanSquaredError, valMeanSquaredError, meanAbsoulteError, valMeanAbsoluteError))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO AnnTraining VALUES (?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def check_mass(self, simulationId, expectedCount):
        """
        Check the number of entries in the table mass with the given simulationId
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert expectedCount is None or type(expectedCount) is int and 0 <= expectedCount

        sqlcommand = 'SELECT COUNT(*) FROM Mass WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckMass: Expected: {} Get: {}***'.format(expectedCount, count[0][0]))

        return count[0][0] == expectedCount


    def insert_mass(self, simulationId, year, massRatio, massDifference, overwrite=False):
        """
        Insert the mass values
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(massRatio) is float and massRatio >= 0
        assert type(massDifference) is float

        sqlcommand_select = 'SELECT simulationId, year FROM Mass WHERE simulationId = ? AND year = ?'
        if overwrite:
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            if len(dataset) != 0:
                sqlcommand = 'DELETE FROM Mass WHERE simulationId = ? AND year = ?'
                self._c.execute(sqlcommand, (simulationId, year))
        else:
            #Test, if dataset for this simulationId and year combination exists
            self._c.execute(sqlcommand_select, (simulationId, year))
            dataset = self._c.fetchall()
            assert len(dataset) == 0
        #Generate and insert spin-up value
        purchases = []
        purchases.append((simulationId, year, massRatio, massDifference))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO Mass VALUES (?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_mass(self, simulationId):
        """
        Delete the data sets of the mass values for the given simulationId.
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'DELETE FROM Mass WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        self._conn.commit()


    def check_tracer_deviation(self, simulationId, expectedCount):
        """
        Check the number of entries of the deviation values with the given simulationId.
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM DeviationTracer WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerDeviation: Expected: {} Get: {}***'.format(expectedCount, count[0][0]))

        return count[0][0] == expectedCount


    def insert_deviation_tracer_tuple(self, simulationId, year, N_mean, N_var, N_min, N_max, N_negative_count, N_negative_sum, DOP_mean=None, DOP_var=None, DOP_min=None, DOP_max=None, DOP_negative_count=None, DOP_negative_sum=None, P_mean=None, P_var=None, P_min=None, P_max=None, P_negative_count=None, P_negative_sum=None, Z_mean=None, Z_var=None, Z_min=None, Z_max=None, Z_negative_count=None, Z_negative_sum=None, D_mean=None, D_var=None, D_min=None, D_max=None, D_negative_count=None, D_negative_sum=None, overwrite=False):
        """
        Insert deviation for the tracer values.
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(N_mean) is float and type(N_var) is float and type(N_min) is float and type(N_max) is float and type(N_negative_count) is float and type(N_negative_sum) is float
        assert (DOP_mean is None and DOP_var is None and DOP_min is None and DOP_max is None and DOP_negative_count is None and DOP_negative_sum is None) or (type(DOP_mean) is float and type(DOP_var) is float and type(DOP_min) is float and type(DOP_max) is float and type(DOP_negative_count) is float and type(DOP_negative_sum) is float)
        assert (P_mean is None and P_var is None and P_min is None and P_max is None and P_negative_count is None and P_negative_sum is None) or (type(P_mean) is float and type(P_var) is float and type(P_min) is float and type(P_max) is float and type(P_negative_count) is float and type(P_negative_sum) is float)
        assert (Z_mean is None and Z_var is None and Z_min is None and Z_max is None and Z_negative_count is None and Z_negative_sum is None) or (type(Z_mean) is float and type(Z_var) is float and type(Z_min) is float and type(Z_max) is float and type(Z_negative_count) is float and type(Z_negative_sum) is float)
        assert (D_mean is None and D_var is None and D_min is None and D_max is None and D_negative_count is None and D_negative_sum is None) or (type(D_mean) is float and type(D_var) is float and type(D_min) is float and type(D_max) is float and type(D_negative_count) is float and type(D_negative_sum) is float)
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
        purchases.append((simulationId, year, N_mean, N_var, N_min, N_max, N_negative_count, N_negative_sum, DOP_mean, DOP_var, DOP_min, DOP_max, DOP_negative_count, DOP_negative_sum, P_mean, P_var, P_min, P_max, P_negative_count, P_negative_sum, Z_mean, Z_var, Z_min, Z_max, Z_negative_count, Z_negative_sum, D_mean, D_var, D_min, D_max, D_negative_count, D_negative_sum))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO DeviationTracer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_deviation_tracer(self, simulationId):
        """
        Delete deviation data sets for the given simulationId.
        @author: Markus Pfeil
        """
        assert type(simulationId) is int and simulationId >= 0

        sqlcommand = 'DELETE FROM DeviationTracer WHERE simulationId = ?'
        self._c.execute(sqlcommand, (simulationId, ))
        self._conn.commit()


    def check_difference_tracer_deviation(self, simulationIdA, simulationIdB, expectedCount):
        """
        Check the number of entries of the deviation values of the difference tracer with the given simulationIdA and simulationIdB
        @author: Markus Pfeil
        """
        assert type(simulationIdA) is int and simulationIdA >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert type(expectedCount) is int and expectedCount >= 0

        sqlcommand = 'SELECT COUNT(*) FROM DeviationTracerDifference WHERE simulationIdA = ? AND simulationIdB = ?'
        self._c.execute(sqlcommand, (simulationIdA, simulationIdB))
        count = self._c.fetchall()
        assert len(count) == 1

        if count[0][0] != expectedCount:
            logging.info('***CheckTracerDifferenceDeviation: Expected: {} Get: {}***'.format(expectedCount, count[0][0]))

        return count[0][0] == expectedCount


    def insert_difference_tracer_deviation_tuple(self, simulationIdA, simulationIdB, yearA, yearB, N_mean, N_var, N_min, N_max, N_negative_count, N_negative_sum, DOP_mean=None, DOP_var=None, DOP_min=None, DOP_max=None, DOP_negative_count=None, DOP_negative_sum=None, P_mean=None, P_var=None, P_min=None, P_max=None, P_negative_count=None, P_negative_sum=None, Z_mean=None, Z_var=None, Z_min=None, Z_max=None, Z_negative_count=None, Z_negative_sum=None, D_mean=None, D_var=None, D_min=None, D_max=None, D_negative_count=None, D_negative_sum=None, overwrite=False):
        """
        Insert deviation for the difference tracer values.
        @author: Markus Pfeil
        """
        assert type(simulationIdA) is int and simulationIdA >= 0
        assert type(simulationIdB) is int and simulationIdB >= 0
        assert type(yearA) is int and yearA >= 0
        assert type(yearB) is int and yearB >= 0
        assert type(N_mean) is float and type(N_var) is float and type(N_min) is float and type(N_max) is float and type(N_negative_count) is float and type(N_negative_sum) is float
        assert (DOP_mean is None and DOP_var is None and DOP_min is None and DOP_max is None and DOP_negative_count is None and DOP_negative_sum is None) or (type(DOP_mean) is float and type(DOP_var) is float and type(DOP_min) is float and type(DOP_max) is float and type(DOP_negative_count) is float and type(DOP_negative_sum) is float)
        assert (P_mean is None and P_var is None and P_min is None and P_max is None and P_negative_count is None and P_negative_sum is None) or (type(P_mean) is float and type(P_var) is float and type(P_min) is float and type(P_max) is float and type(P_negative_count) is float and type(P_negative_sum) is float)
        assert (Z_mean is None and Z_var is None and Z_min is None and Z_max is None and Z_negative_count is None and Z_negative_sum is None) or (type(Z_mean) is float and type(Z_var) is float and type(Z_min) is float and type(Z_max) is float and type(Z_negative_count) is float and type(Z_negative_sum) is float)
        assert (D_mean is None and D_var is None and D_min is None and D_max is None and D_negative_count is None and D_negative_sum is None) or (type(D_mean) is float and type(D_var) is float and type(D_min) is float and type(D_max) is float and type(D_negative_count) is float and type(D_negative_sum) is float)
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
        purchases.append((simulationIdA, simulationIdB, yearA, yearB, N_mean, N_var, N_min, N_max, N_negative_count, N_negative_sum, DOP_mean, DOP_var, DOP_min, DOP_max, DOP_negative_count, DOP_negative_sum, P_mean, P_var, P_min, P_max, P_negative_count, P_negative_sum, Z_mean, Z_var, Z_min, Z_max, Z_negative_count, Z_negative_sum, D_mean, D_var, D_min, D_max, D_negative_count, D_negative_sum))

        inserted = False
        insertCount = 0
        while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
            try:
                self._c.executemany('INSERT INTO DeviationTracerDifference VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
                self._conn.commit()
                inserted = True
            except sqlite3.OperationalError:
                insertCount += 1
                #Wait for the next insert
                time.sleep(DB_Constants.TIME_SLEEP)


    def delete_difference_tracer_deviation(self, simulationIdA):
        """
        Delete data sets of the deviation for the difference tracer values for the given simulationId.
        @author: Markus Pfeil
        """
        assert type(simulationIdA) is int and simulationIdA >= 0

        sqlcommand = 'DELETE FROM DeviationTracerDifference WHERE simulationIdA = ?'
        self._c.execute(sqlcommand, (simulationIdA, ))
        self._conn.commit()


    #Functions to read values from the database
    def read_losses(self, annId):
        """
        Read losses (loss and valLoss) for the given annId.
        @author: Markus Pfeil
        """
        assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)

        sqlcommand = 'SELECT epoch, loss, valLoss FROM AnnTraining WHERE annId = ? ORDER BY epoch;'
        self._c.execute(sqlcommand, (annId, ))
        rows = self._c.fetchall()
        data = np.empty(shape=(len(rows), 3))

        i = 0
        for row in rows:
            data[i,:] = np.array([row[0], row[1], row[2]])
            i = i+1
        return data


    def read_rel_norm(self, model, concentrationId, massCorrection=False, tolerance=None, cpus=64, year=None, norm='2', parameterId=None, trajectory=''):
        """
        Read for every parameterId the norm value for the given annId from the database.
        If parameterId is not None, read only the relative difference for the given parameterId.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(concentrationId) is int and concentrationId >= 0
        assert type(massCorrection) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert parameterId is None or type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert year is None or type(year) is int and year >= 0
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        parameterIdStr = '' if parameterId is None else 'AND sim.parameterId = {:d} '.format(parameterId)
        toleranceStr = 'sim.tolerance is NULL' if tolerance is None else 'sim.tolerance = ?'
        if year is None:
            yearStr = 'NOT EXISTS (SELECT * FROM TracerDifference{}Norm AS diff2 WHERE diff.simulationIdA = diff2.simulationIdA AND diff.simulationIdB = diff2.simulationIdB AND diff.yearB = diff2.yearB AND diff.yearA < diff2.yearA)'.format(norm)
        else:
            yearStr = 'diff.yearA = {:d}'.format(year)

        sqlcommand = 'SELECT sim.simulationId AS SimulationId, diff.tracer/refnorm.tracer AS Error FROM Simulation AS sim, Simulation AS simref, TracerDifference{}{}Norm AS diff, Tracer{}{}Norm AS refnorm WHERE sim.model = ? {}AND sim.concentrationId = ? AND sim.massCorrection = ? AND {} AND sim.cpus = ? AND simref.model = sim.model AND simref.parameterId = sim.parameterId AND simref.concentrationId = ? AND simref.massCorrection = ? AND simref.tolerance is NULL AND simref.cpus = ? AND diff.simulationIdA = sim.simulationId AND diff.simulationIdB = simref.simulationId AND diff.yearB = ? AND refnorm.simulationId = simref.simulationId AND refnorm.year = diff.yearB AND {} ORDER BY sim.simulationId'.format(trajectory, norm, trajectory, norm, parameterIdStr, toleranceStr, yearStr)

        if tolerance is None:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), cpus, ANN_DB_Constants.CONCENTRATIONID_CONSTANT[model], 0, 64, 10000))
        else:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), tolerance, cpus, ANN_DB_Constants.CONCENTRATIONID_CONSTANT[model], 0, 64, 10000))
        normValues = self._c.fetchall()
        normdata = np.empty(shape=(len(normValues)))

        i = 0
        for row in normValues:
            normdata[i] = row[1]
            i = i+1

        return normdata


    def read_mass(self, model, concentrationId, massCorrection=False, tolerance=None, cpus=64, year=0):
        """
        Read the mass ratio (mass divided by mass of the initial concentration) for every parameterId for the given ann setting.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(concentrationId) is int and concentrationId >= 0
        assert type(massCorrection) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert year is None or type(year) is int and year >= 0

        toleranceStr = 'sim.tolerance is NULL' if tolerance is None else 'sim.tolerance = ?'
        if year is None:
            yearStr = 'NOT EXISTS (SELECT * FROM Mass AS mass2 WHERE mass.simulationId = mass2.simulationId AND mass.year < mass2.year)'
        else:
            yearStr = 'mass.year = {:d}'.format(year)

        sqlcommand = 'SELECT mass.simulationId AS SimulationId, mass.ratio AS MassRation FROM Simulation AS sim, Mass AS mass WHERE sim.model = ? AND sim.concentrationId = ? AND sim.massCorrection = ? AND {} AND sim.cpus = ? AND mass.simulationId = sim.simulationId AND {} ORDER BY mass.simulationId'.format(toleranceStr, yearStr)

        if tolerance is None:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), cpus))
        else:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), tolerance, cpus))
        massValues = self._c.fetchall()
        massData = np.empty(shape=(len(massValues)))

        i = 0
        for row in massValues:
            massData[i] = row[1]
            i = i+1

        return massData


    def read_spinup_tolerance(self, model, concentrationId, massCorrection=False, tolerance=None, cpus=64, year=0):
        """
        Read the spin up tolerance for every parameterId of the given year for the given ann setting.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(concentrationId) is int and concentrationId >= 0
        assert type(massCorrection) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert year is None or type(year) is int and year >= 0

        toleranceStr = 'sim.tolerance is NULL' if tolerance is None else 'sim.tolerance = ?'
        if year is None:
            yearStr = 'NOT EXISTS (SELECT * FROM Spinup AS sp2 WHERE sp.simulationId = sp2.simulationId AND sp.year < sp2.year)'
        else:
            yearStr = 'sp.year = {:d}'.format(year)

        sqlcommand = 'SELECT sp.simulationId AS SimulationId, sp.tolerance AS Tolerance FROM Simulation AS sim, Spinup AS sp WHERE sim.model = ? AND sim.concentrationId = ? AND sim.massCorrection = ? AND {} AND sim.cpus = ? AND sp.simulationId = sim.simulationId AND {} ORDER BY sp.simulationId'.format(toleranceStr, yearStr)

        if tolerance is None:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), cpus))
        else:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), tolerance, cpus))
        spinupValues = self._c.fetchall()
        spinupData = np.empty(shape=(len(spinupValues)))

        i = 0
        for row in spinupValues:
            spinupData[i] = row[1]
            i = i+1

        return spinupData


    def read_spinup_year(self, model, concentrationId, massCorrection=False, tolerance=None, cpus=64):
        """
        Read the required years to reach the given spin up tolerance for every parameterId for the given ann setting.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(concentrationId) is int and concentrationId >= 0
        assert type(massCorrection) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0

        toleranceStr = 'sim.tolerance is NULL' if tolerance is None else 'sim.tolerance = ?'

        sqlcommand = 'SELECT sp.simulationId AS SimulationId, sim.parameterId AS ParameterId, sp.year AS Year FROM Simulation AS sim, Spinup AS sp WHERE sim.model = ? AND sim.concentrationId = ? AND sim.massCorrection = ? AND {} AND sim.cpus = ? AND sp.simulationId = sim.simulationId AND NOT EXISTS (SELECT * FROM Spinup AS sp2 WHERE sp.simulationId = sp2.simulationId AND sp.year < sp2.year) ORDER BY sp.simulationId'.format(toleranceStr)

        if tolerance is None:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), cpus))
        else:
            self._c.execute(sqlcommand, (model, concentrationId, int(massCorrection), tolerance, cpus))
        spinupValues = self._c.fetchall()
        spinupData = np.empty(shape=(2, len(spinupValues)))

        i = 0
        for row in spinupValues:
            spinupData[0, i] = row[1]
            spinupData[1, i] = row[2]
            i = i+1

        return spinupData
