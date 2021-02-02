#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os

from metos3dutil.database.AbstractClassDatabaseMetos3d import AbstractClassDatabaseMetos3d
import sbo.constants as SBO_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants


class SBO_Database(AbstractClassDatabaseMetos3d):
    """
    Class for the database access.
    @author: Markus Pfeil
    """

    def __init__(self, dbpath=SBO_Constants.DB_PATH):
        """
        Initialization of the database connection
        @author: Markus Pfeil
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)

        AbstractClassDatabaseMetos3d.__init__(self, dbpath)


    def get_annTypeModel(self, annId):
        """
        Get the annType and the model for a given annId
        @author: Markus Pfeil
        """
        assert annId in range(0, SBO_Constants.ANNID_MAX + 1)
        
        sqlcommand = 'SELECT annType, model FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1])


    def get_annTypeNumberModel(self, annId):
        """
        Get the annType, the annNumber and the model for a given annId
        @author: Markus Pfeil
        """
        assert annId in range(0, SBO_Constants.ANNID_MAX + 1)
        
        sqlcommand = 'SELECT annType, annNumber, model FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1], annConfig[0][2])


    def get_annConfig(self, annId):
        """
        Get the parameter of the ann configuration for the given annId
        @author: Markus Pfeil
        """
        assert annId in range(0, SBO_Constants.ANNID_MAX + 1)
        
        sqlcommand = 'SELECT annType, annNumber, model, epochs, trainingData FROM AnnConfig WHERE annId = ?'
        self._c.execute(sqlcommand, (annId,))
        annConfig = self._c.fetchall()
        assert len(annConfig) == 1
        return (annConfig[0][0], annConfig[0][1], annConfig[0][2], annConfig[0][3], annConfig[0][4])


    def insert_annConfig(self, annType, annNumber, model, epochs, trainingData):
        """
        Insert the configuration of an ann
        @author: Markus Pfeil
        """
        assert type(annType) is str
        assert type(annNumber) is int and 0 <= annNumber
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(epochs) is int and 0 < epochs
        assert type(trainingData) is int and 0 < trainingData
        
        sqlcommand = 'SELECT MAX(annId) FROM AnnConfig'
        self._c.execute(sqlcommand)
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        annId = dataset[0][0] + 1
        
        purchases = []
        purchases.append((annId, annType, annNumber, model, epochs, trainingData))
        self._c.executemany('INSERT INTO AnnConfig VALUES (?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def exists_optimization(self, optimizationId):
        """
        Exists an entry for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT COUNT(*) FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        optimization = self._c.fetchall()
        assert len(optimization) == 1 and (optimization[0][0] == 0 or optimization[0][0] == 1)
        return optimization[0][0] == 1


    def get_model(self, optimizationId):
        """
        Get the model for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT model FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        model = self._c.fetchall()
        assert len(model) == 1
        return model[0][0]


    def get_target_parameter(self, optimizationId):
        """
        Get the model parameter of the target parameter for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT model, parameterId FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        
        parameter = self.get_parameter(dataset[0][1], dataset[0][0])
        return parameter


    def get_initial_parameter(self, optimizationId):
        """
        Get the model parameter of the initial parameter for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT model, initialParameterId FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        
        parameter = self.get_parameter(dataset[0][1], dataset[0][0])
        return parameter


    def get_highFidelityModel(self, optimizationId):
        """
        Get the model years and the timestep used for the high fidelity model for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT highFidelityModelYears, highFidelityModelTimestep FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (dataset[0][0], dataset[0][1])


    def get_lowFidelityModel(self, optimizationId):
        """
        Get the model years and the timestep used for the low fidelity model for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT lowFidelityModelYears, lowFidelityModelTimestep FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (dataset[0][0], dataset[0][1])


    def usesLowFidelityModelAnn(self, optimizationId):
        """
        Uses the low fidelity model an artificial neural model or metos3d
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT lowFidelityModelAnnId FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return dataset[0][0] is not None


    def get_annConfigLowFidelityModel(self, optimizationId):
        """
        Read the configuration of the artificial neural network used as low fidelity model
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT ann.annType, ann.annNumber FROM AnnConfig AS ann, Optimization AS opt WHERE opt.optimizationId = ? AND opt.lowFidelityModelAnnId = ann.annId'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (dataset[0][0], dataset[0][1])
    
    
    def get_terminationCondition(self, optimizationId):
        """
        Get the parameter of the termination condition
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT gamma, delta, maxIteration, maxIterationOptimization FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (dataset[0][0], dataset[0][1], dataset[0][2], dataset[0][3])


    def get_surrogateCorrectionParameter(self, optimizationId):
        """
        Get the correction parameter of the surrogate
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT surrogateEnhancement, surrogateLowerBound, surrogateUpperBound, surrogateThreshold FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (bool(dataset[0][0]), dataset[0][1], dataset[0][2], dataset[0][3])


    def get_trajectoryNorm(self, optimizationId):
        """
        Get the trajectory norm flag for the cost function norm
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT trajectoryNorm FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return bool(dataset[0][0])


    def get_trustRegionRadiusParameter(self, optimizationId):
        """
        Read the parameter of the trust region radius
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId

        sqlcommand = 'SELECT trustRegionRadius, trustRegionRadiusMincr, trustRegionRadiusMdecr, trustRegionRadiusRincr, trustRegionRadiusRdecr FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return (dataset[0][0], dataset[0][1], dataset[0][2], dataset[0][3], dataset[0][4])


    def get_method(self, optimizationId):
        """
        Read the method of the optimization
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT method FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        return dataset[0][0]


    def insert_optimization(self, model, parameterId, initialParameterId, highFidelityModelYears = 5000, highFidelityModelTimestep = 64, lowFidelityModelAnnId = 0, lowFidelityModelYears = None, lowFidelityModelTimestep = None, gamma = 5 * 10**(-2), delta = 5 * 10**(-3), maxIterations = 10, maxIterationOptimization = None, surrogateEnhancement = True, surrogateLowerBound = 0.1, surrogateUpperBound = 5.0, surrogateThreshold = 5 * 10**(-3), trajectoryNorm = True, method = 'L-BFGS-B', cpus = NeshCluster_Constants.DEFAULT_NODES * NeshCluster_Constants.CORES):
        """
        Insert the configuration of an optimization
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and 0 <= parameterId
        assert type(initialParameterId) is int and 0 <= initialParameterId
        assert type(highFidelityModelYears) is int and 0 < highFidelityModelYears
        assert highFidelityModelTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert lowFidelityModelAnnId is None or (type(lowFidelityModelAnnId) is int and 0 <= lowFidelityModelAnnId)
        assert lowFidelityModelYears is None or (type(lowFidelityModelYears) is int and 0 < lowFidelityModelYears)
        assert lowFidelityModelTimestep is None or lowFidelityModelTimestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert (lowFidelityModelAnnId is None and lowFidelityModelYears is not None and lowFidelityModelTimestep is not None) or (lowFidelityModelAnnId is not None and lowFidelityModelYears is None and lowFidelityModelTimestep is None)
        assert type(gamma) is float and 0 < gamma
        assert type(delta) is float and 0 < delta
        assert type(maxIterations) is int and 0 < maxIterations
        assert maxIterationOptimization is None or (type(maxIterationOptimization) is int and 0 < maxIterationOptimization)
        assert type(surrogateEnhancement) is bool
        assert (not surrogateEnhancement and surrogateLowerBound is None) or (surrogateEnhancement and type(surrogateLowerBound) is float and 0 < surrogateLowerBound)
        assert (not surrogateEnhancement and surrogateUpperBound is None) or (surrogateEnhancement and type(surrogateUpperBound) is float and 0 < surrogateUpperBound)
        assert (not surrogateEnhancement and surrogateThreshold is None) or (surrogateEnhancement and type(surrogateThreshold) is float and 0 < surrogateThreshold)
        assert type(trajectoryNorm) is bool
        assert method in SBO_Constants.OPTIMIZATION_METHODS
        assert type(cpus) is int and 0 < cpus
        
        sqlcommand = 'SELECT COUNT(*) FROM Parameter WHERE parameterId = ?'
        self._c.execute(sqlcommand, (parameterId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        assert dataset[0][0] == 1
        
        self._c.execute(sqlcommand, (initialParameterId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        assert dataset[0][0] == 1
        
        if lowFidelityModelAnnId is not None:
            sqlcommand = 'SELECT COUNT(*) FROM AnnConfig WHERE annId = ?'
            self._c.execute(sqlcommand, (lowFidelityModelAnnId,))
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            assert dataset[0][0] == 1
        
        sqlcommand = 'SELECT MAX(optimizationId) FROM Optimization'
        self._c.execute(sqlcommand)
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        optimizationId = dataset[0][0] + 1
        
        purchases = []
        purchases.append((optimizationId, model, parameterId, initialParameterId, highFidelityModelYears, highFidelityModelTimestep, lowFidelityModelAnnId, lowFidelityModelYears, lowFidelityModelTimestep, gamma, delta, maxIterations, maxIterationOptimization, int(surrogateEnhancement), surrogateLowerBound, surrogateUpperBound, surrogateThreshold, int(trajectoryNorm), method, cpus))
        self._c.executemany('INSERT INTO Optimization VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def exists_iteration(self, optimizationId):
        """
        Exists an entry for the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        
        sqlcommand = 'SELECT COUNT(*) FROM Iteration WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        iteration = self._c.fetchall()
        assert len(iteration) == 1
        return iteration[0][0] > 0


    def insert_iteration(self, optimizationId, iteration, trustRegionRadiusIteration, accepted, parameterId, stepSizeNorm, costfunctionValueHighFidelityModel, trustRegionRadius, costfunctionValueOptimization, numberOfIterations, numberOfFunctionEvaluations, time):
        """
        Insert the dataset for a iteration
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        assert type(iteration) is int and 0 <= iteration
        assert type(trustRegionRadiusIteration) is int and 0 <= trustRegionRadiusIteration
        assert type(accepted) is bool
        assert type(parameterId) is int and 0 <= parameterId
        assert type(stepSizeNorm) in [float, np.float64, np.int64] and 0 <= stepSizeNorm
        assert type(costfunctionValueHighFidelityModel) in [float, np.float64, np.int64] and 0 <= costfunctionValueHighFidelityModel
        assert type(trustRegionRadius) is float and 0 < trustRegionRadius
        assert type(costfunctionValueOptimization) in [float, np.float64, np.int64] and 0 <= costfunctionValueOptimization
        assert type(numberOfIterations) is int and 0 <= numberOfIterations
        assert type(numberOfFunctionEvaluations) is int and 0 <= numberOfFunctionEvaluations
        assert type(time) is float and 0 <= time
        
        sqlcommand = 'SELECT COUNT(*) FROM Optimization WHERE optimizationId = ?'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        assert dataset[0][0] == 1
        
        sqlcommand = 'SELECT COUNT(*) FROM Parameter WHERE parameterId = ?'
        self._c.execute(sqlcommand, (parameterId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1
        assert dataset[0][0] == 1
        
        purchases = []
        purchases.append((optimizationId, iteration, trustRegionRadiusIteration, int(accepted), parameterId, float(stepSizeNorm), float(costfunctionValueHighFidelityModel), trustRegionRadius, float(costfunctionValueOptimization), numberOfIterations, numberOfFunctionEvaluations, time))
        self._c.executemany('INSERT INTO Iteration VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
        self._conn.commit()


    def get_costfunctionValue(self, optimizationId, costfunctionModel='HighFidelityModel', acceptedOnly=True):
        """
        Get the cost function values for the SBO optimization run
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        assert costfunctionModel in ['HighFidelityModel', 'Optimization']
        assert type(acceptedOnly) is bool

        if acceptedOnly:
            sqlcommand = 'SELECT iteration, costfunctionValue{} FROM Iteration WHERE optimizationId = ? AND accepted = 1 ORDER BY iteration'.format(costfunctionModel)
        else:
            sqlcommand = 'SELECT iteration, costfunctionValue{} FROM Iteration WHERE optimizationId = ? ORDER BY iteration, trustRegionRadiusIteration'.format(costfunctionModel)

        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        costfunctionValues = np.empty(shape = (2, len(dataset)))

        i = 0
        for row in dataset:
            costfunctionValues[0, i] = row[0]
            costfunctionValues[1, i] = row[1]
            i = i + 1

        return costfunctionValues
    

    def get_stepsizeNormValue(self, optimizationId):
        """
        Get the step size norm for the SBO optimization run
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId

        sqlcommand = 'SELECT iteration, stepSizeNorm FROM Iteration WHERE optimizationId = ? AND accepted = 1 ORDER BY iteration'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        stepsizeNormValues = np.empty(shape = (2, len(dataset)))

        i = 0
        for row in dataset:
            stepsizeNormValues[0, i] = row[0]
            stepsizeNormValues[1, i] = row[1]
            i = i + 1

        return stepsizeNormValues


    def get_iteration_parameterIds(self, optimizationId):
        """
        Get the parameter ids for the SBO optimization run
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId

        sqlcommand = 'SELECT iteration, parameterId FROM Iteration WHERE optimizationId = ? AND accepted = 1 ORDER BY iteration'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        parameterIdValues = []

        for row in dataset:
            parameterIdValues.append((row[0], row[1]))

        return parameterIdValues


    def get_count_iterations(self, optimizationId):
        """
        Count the number of accepted iteration of the SBO optimization run
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId

        sqlcommand = 'SELECT COUNT(*) FROM Iteration WHERE optimizationId = ? AND accepted = 1'
        self._c.execute(sqlcommand, (optimizationId,))
        dataset = self._c.fetchall()
        assert len(dataset) == 1

        return dataset[0][0]


    def get_parameter_iteration(self, optimizationId, iteration):
        """
        Get the parameter for the iteration of the SBO optimization run
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        assert type(iteration) is int and 0 <= iteration

        sqlcommand = 'SELECT opt.model, it.parameterId FROM Optimization AS opt, Iteration AS it WHERE opt.optimizationId = ? AND opt.optimizationId = it.optimizationId AND it.iteration = ? AND it.accepted = 1'
        self._c.execute(sqlcommand, (optimizationId, iteration))
        dataset = self._c.fetchall()
        assert len(dataset) == 1

        parameter = self.get_parameter(dataset[0][1], dataset[0][0])

        return parameter

