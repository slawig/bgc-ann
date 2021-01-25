#!/usr/bin/env python
# -*- coding: utf8 -*

import bz2
import errno
import logging
import numpy as np
import os
import scipy.optimize
import shutil
import tarfile
import time

import metos3dutil.metos3d.Metos3d as Metos3d
import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants
from sbo.AbstractClassSurrogateBasedOptimization import AbstractClassSurrogateBasedOptimization
import sbo.constants as SBO_Constants
from plot.SurrogateBasedOptimizationPlot import SurrogateBasedOptimizationPlot


class SurrogateBasedOptimization(AbstractClassSurrogateBasedOptimization):
    """
    @author: Markus Pfeil
    """

    def __init__(self, optimizationId, queue=NeshCluster_Constants.DEFAULT_QUEUE, cores=NeshCluster_Constants.DEFAULT_CORES):
        """
        Initialisation of the surrogate based optimization
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        assert queue in NeshCluster_Constants.QUEUE
        assert type(cores) is int and 0 < cores

        AbstractClassSurrogateBasedOptimization.__init__(self, optimizationId, queue=queue, cores=cores)
        
        #Parameter for the target concentration calculation
        self._targetTracerModelYears = SBO_Constants.TARGET_TRACER_MODEL_YEARS
        self._targetTracerModelTimestep = SBO_Constants.TARGET_TRACER_MODEL_TIMESTEP
        self._setTargetModelParameter(list(self._sboDB.get_target_parameter(self._optimizationId)))
        self._targetTracerConcentration = None
        
        self._modelParameter = {}
        self._k = 0
        self._j = 0
        
        #Initial model parameter for the optimization
        self._setInitialModelParameter(list(self._sboDB.get_initial_parameter(self._optimizationId)))
       
        #Set parameter for the termination of the optimization 
        (gamma, delta, maxIterations, maxIterationOptimization) = self._sboDB.get_terminationCondition(self._optimizationId)
        self._setTerminationCondition(gamma = gamma, delta = delta, maxIterations = maxIterations, maxIterationOptimization = maxIterationOptimization)
        
        #Set parameter for the trust region radius
        (trustRegionRadius, mincr, mdecr, rincr, rdecr) = self._sboDB.get_trustRegionRadiusParameter(self._optimizationId)
        self._setTrustRegionRadius(trustRegionRadius = trustRegionRadius, mincr = mincr, mdecr = mdecr, rincr = rincr, rdecr = rdecr)

        self._setOptimizationMethod(self._sboDB.get_method(self._optimizationId))       
 

    def _setTargetTracerModel(self, modelYears = 10001, timestep = 1):
        """
        Set the parameter to calculate the target concentration
        @author: Markus Pfeil
        """
        assert type(modelYears) is int and 0 < modelYears and self._highFidelityModelYears <= modelYears
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS and self._highFidelityModelTimestep <= timestep
        
        self._targetTracerModelYears = modelYears
        self._targetTracerModelTimestep = timestep
        logging.info('****Set parameter for the target concentration calculation****\nModel years: {:d}\nTimestep: {:d}'.format(self._targetTracerModelYears, self._targetTracerModelTimestep))


    def _setTerminationCondition(self, gamma = 5 * 10**(-2), delta = 5 * 10**(-3), maxIterations = 10, maxIterationOptimization = None):
        """
        Set the parameter for the termination conditions
        @author: Markus Pfeil
        """
        assert type(gamma) is float and 0 < gamma
        assert type(delta) is float and 0 < delta
        assert type(maxIterations) is int and 0 < maxIterations
        assert maxIterationOptimization is None or (type(maxIterationOptimization) is int and 0 < maxIterationOptimization)
        
        self._terminationConditionGamma = gamma
        self._terminationConditionDelta = delta
        self._maxIterationOptimizeSurrogate = maxIterations
        self._terminationConditionMaxIteration = maxIterationOptimization
        logging.info('****Set parameter for the termination condition****\nGamma: {:e}\nDelta: {:e}\nMaxIterations: {:d}\nMaxIterationOptimization: {}'.format(self._terminationConditionGamma, self._terminationConditionDelta, self._maxIterationOptimizeSurrogate, self._terminationConditionMaxIteration))


    def _setTrustRegionRadius(self, trustRegionRadius = 6 * 10**(-2), mincr = 3.0, mdecr = 20.0, rincr = 0.75, rdecr = 0.01):
        """
        Set the parameter to calculate the update of the trust region radius
        @author: Markus Pfeil
        """
        assert type(trustRegionRadius) is float and 0 < trustRegionRadius
        assert type(mincr) is float and 0 < mincr
        assert type(mdecr) is float and 0 < mdecr
        assert type(rincr) is float and 0 < rincr
        assert type(rdecr) is float and 0 < rdecr

        self._trustRegionRadius = trustRegionRadius
        self._trustRegionRadiusIncreaseConstant = mincr
        self._trustRegionRadiusDecreaseConstant = mdecr
        self._trustRegionRadiusIncreaseBound = rincr
        self._trustRegionRadiusDecreaseBound = rdecr
        logging.info('****Set parameter for the trust region radius calculation****\nTrustRegionRadius: {:e}\nm_incr: {:e}\nm_decr: {:e}\nr_incr: {:e}\nr_decr: {:e}'.format(self._trustRegionRadius, self._trustRegionRadiusIncreaseConstant, self._trustRegionRadiusDecreaseConstant, self._trustRegionRadiusIncreaseBound, self._trustRegionRadiusDecreaseBound))


    def _setInitialModelParameter(self, modelParameter):
        """
        Set the intial model parameter
        @author: Markus Pfeil
        """
        assert type(modelParameter) is list and len(modelParameter) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]
        assert np.all(np.array(Metos3d_Constants.LOWER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]]) <= np.array(modelParameter))
        assert np.all(np.array(modelParameter) <= np.array(Metos3d_Constants.UPPER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]]))
        
        self._initialParameter = np.array(modelParameter)
        logging.info('****Set initial model parameter****\nInitial model parameter: {}'.format(self._initialParameter))


    def _setTargetModelParameter(self, modelParameter):
        """
        Set the target model parameter
        @author: Markus Pfeil
        """
        assert type(modelParameter) is list and len(modelParameter) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]
        
        self._targetModelParameter = modelParameter
        logging.info('****Set target model parameter****\nTarget model parameter: {}'.format(self._targetModelParameter))


    def _setOptimizationMethod(self, method):
        """
        @author: Markus Pfeil
        """
        assert method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
        self._optimizationMethod = method
        logging.info('****Set the optimization method to {}****'.format(self._optimizationMethod))


    def run(self, options = None):
        """
        Start the surrogate based optimization
        @author: Markus Pfeil
        """
        assert options is None or type(options) is dict
        
        timeStart = time.time()
        
        if options is None: 
            options = {'maxiter': self._maxIterationOptimizeSurrogate, 'disp': True}
        
        logging.info('****Start the surrogate based optimization****\nMethod: {}\nOptions: {}'.format(self._optimizationMethod, options))
        
        #Initialisation
        accepted = True
        iterationAccepted = 0
        self._setPath()
        self._k = 0
        self._j = 0
        self._modelParameter[self._k] = list(self._initializeParameter())
        
        if self._targetModelParameter is not None and self._targetTracerConcentration is None:
            self._calculateTargetTracerConcentration()
        
        tracerHighFidelityModel = self._highFidelityModel(self._modelParameter[self._k])
        costfunctionValueHighFidelityModel = self._misfit(tracerHighFidelityModel, highFidelityModel = True)
        
        parameterId = self._sboDB.insert_parameter(self._modelParameter[self._k], self._model)
        self._sboDB.insert_iteration(self._optimizationId, self._k, iterationAccepted, accepted, parameterId, 0.0, costfunctionValueHighFidelityModel, self._trustRegionRadius, 0.0, 0, 0, 0.0)
        
        while (self._k == 0 and self._trustRegionRadius >= self._terminationConditionDelta) or (self._k > 0 and not self._terminationConditionSBO()):
            timeStartIteration = time.time()
            logging.debug('Optimization iteration: {:d}\nModel parameter: {}'.format(self._k, self._modelParameter[self._k]))
            
            if accepted:
                tracerLowFidelityModel = self._lowFidelityModel(self._modelParameter[self._k])
                timeLowFidelityModel = time.time()

                tracerHighFidelityModelRestricted = tracerHighFidelityModel
                if self._useTrajectoryNorm:
                    tracerHighFidelityModelRestricted = tracerHighFidelityModelRestricted[::self._lowFidelityModelTimestep//self._highFidelityModelTimestep]
                
                self._constructSurrogate(tracerHighFidelityModelRestricted, tracerLowFidelityModel)
                timeConstructSurrogate = time.time()
            
            #Optimization using the surrogate
            #Using bounds for the trust region radius (corresponding to the max-Norm)
            bounds = scipy.optimize.Bounds(np.maximum(np.array(self._modelParameter[self._k]) - self._trustRegionRadius, Metos3d_Constants.LOWER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]]), np.minimum(np.array(self._modelParameter[self._k]) + self._trustRegionRadius, Metos3d_Constants.UPPER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]]))
            logging.debug('***Bounds: {}'.format(bounds))
            logging.debug('****Start optimization using method {}****'.format(self._optimizationMethod))
            res = scipy.optimize.minimize(lambda u: self._objective(u), self._modelParameter[self._k], method=self._optimizationMethod, bounds=bounds, options=options)
            logging.info('****Optimation result of iteration {:d} using the surrogate****\n{}'.format(self._k, res))
            timeOptimization = time.time()
            
            self._k = self._k + 1
            self._modelParameter[self._k] = list(res.x)

            tracerHighFidelityModel_new = self._highFidelityModel(self._modelParameter[self._k])
            costfunctionValueHighFidelityModel_new = self._misfit(tracerHighFidelityModel_new, highFidelityModel = True)
            timeHighFidelityModel = time.time()
            
            logging.debug('****Costfunction values: J(y(u_k+1)) = {:e} and J(y(u_k)) = {:e}****'.format(costfunctionValueHighFidelityModel_new, costfunctionValueHighFidelityModel))

            if costfunctionValueHighFidelityModel_new < costfunctionValueHighFidelityModel:
                accepted = True
                #Decrement k in order to evaluate the low fidelity model in the correct directory
                self._k = self._k - 1
                s_new = self._misfit(self._evaluateSurrogate(self._modelParameter[self._k + 1]))
                s_old = self._misfit(self._evaluateSurrogate(self._modelParameter[self._k]))
                self._k = self._k + 1
                
                assert s_new != s_old
                rho = (costfunctionValueHighFidelityModel_new - costfunctionValueHighFidelityModel) / (s_new - s_old)
                self._updateTrustRegionRadius(rho)
                tracerHighFidelityModel = tracerHighFidelityModel_new
                costfunctionValueHighFidelityModel = costfunctionValueHighFidelityModel_new
                self._j = 0
            else:
                accepted = False

                #Remove directory of the last highFidelityModel
                metos3dSimulationPath = os.path.join(self._path, SBO_Constants.PATH_ITERATION.format(self._k), SBO_Constants.PATH_HIGH_FIDELITY_MODEL)
                if os.path.exists(metos3dSimulationPath) and os.path.isdir(metos3dSimulationPath):
                    shutil.rmtree(metos3dSimulationPath)

                self._decreaseTrustRegionRadius()
            
            timeTrustRegionRadius = time.time()
            
            #Insert optimization data in the database
            parameterId = self._sboDB.insert_parameter(self._modelParameter[self._k], self._model)
            self._sboDB.insert_iteration(self._optimizationId, self._k, iterationAccepted, accepted, parameterId, self._stepSizeNorm(), costfunctionValueHighFidelityModel_new, self._trustRegionRadius, float(res.fun), int(res.nit), int(res.nfev), timeOptimization - timeStartIteration)

            timeDatabase = time.time()

            if accepted:
                iterationAccepted = 0
                logging.debug('****Time for the iteration {:d}: {:f}s****\nTime for low fidelity model: {:f}s\nTime for construction of the surrogate: {:f}s\nTime for the optimization: {:f}s\nTime for high fidelity model: {:f}s\nTime for construction of the new trust region radius: {:f}s\nTime for the database insert: {:f}s****'.format(self._k, timeDatabase - timeStartIteration, timeLowFidelityModel - timeStartIteration, timeConstructSurrogate - timeLowFidelityModel, timeOptimization - timeConstructSurrogate, timeHighFidelityModel - timeOptimization, timeTrustRegionRadius - timeHighFidelityModel, timeDatabase- timeTrustRegionRadius))
            else:
                self._k = self._k - 1
                iterationAccepted = iterationAccepted + 1
                logging.debug('****Time for the iteration {:d}: {:f}s****\nTime for the optimization: {:f}s\nTime for high fidelity model: {:f}s\nTime for construction of the new trust region radius: {:f}s\nTime for the database insert: {:f}s****'.format(self._k, timeDatabase - timeStartIteration, timeOptimization - timeStartIteration, timeHighFidelityModel - timeOptimization, timeTrustRegionRadius - timeHighFidelityModel, timeDatabase- timeTrustRegionRadius))
        
        logging.info('Optimal model parameter: {}\n****End of the surrogate based optimization****'.format(self._modelParameter[self._k]))
        
        self._removeTargetTracerTrajectory()
        
        timeEnd = time.time()
        logging.info('****Overall time: {:f}s****'.format(timeEnd - timeStart))
        
        return self._modelParameter[self._k]


    def _initializeParameter(self):
        """
        Initialisation of the model parameter
        @author: Markus Pfeil
        """
        if self._initialParameter is None:
            logging.debug('***Generate initial model parameter***')
            self._initialParameter = [random.random() for _ in Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]]
            self._initialParameter = Metos3d_Constants.LOWER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]] + (Metos3d_Constants.UPPER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]] - Metos3d_Constants.LOWER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[self._model]]) * self._initialParameter
        
        return self._initialParameter


    def _setPath(self):
        """
        Create the directory for the optimization
        @author: Markus Pfeil
        """
        self._path = os.path.join(SBO_Constants.PATH, 'Optimization', SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId))
        os.makedirs(self._path, exist_ok=False)
        logging.debug('Path for the optimization: {}'.format(self._path))


    def _objective(self, u):
        """
        Calculation the cost function value using the surrogate for the the given model parameter
        This function uses the surrogate to calculate the tracer concentration and compute the cost function value.
        @author: Markus Pfeil
        """
        assert (type(u) is np.ndarray or type(u) is list) and len(u) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]
        
        logging.debug('***Evaluation of the objective function***\nModel parameter: {}'.format(u))
        y = self._evaluateSurrogate(list(u))
        J = self._misfit(np.array(y))
        
        return J


    def _misfit(self, y, highFidelityModel = False):
        """
        Calculate the cost function value
        @author: Markus Pfeil
        """
        assert self._targetTracerConcentration is not None
        assert type(highFidelityModel) is bool
        assert type(y) is np.ndarray and y.shape == self._targetTracerConcentration[::self._highFidelityModelTimestep if highFidelityModel else self._lowFidelityModelTimestep].shape
        
        J = 0.5 * np.sum((y - self._targetTracerConcentration[::self._highFidelityModelTimestep if highFidelityModel else self._lowFidelityModelTimestep])**2)
        logging.debug('Misfit: {:e}'.format(J))
        return J


    def _get_path_lowFidelityModel(self):
        """
        Get the path of the low fidelity model for the current evaluation.
        @author: Markus Pfeil
        """
        path = os.path.join(self._path, SBO_Constants.PATH_ITERATION.format(self._k), SBO_Constants.PATH_LOW_FIDELITY_MODEL.format(self._j))
        assert not os.path.exists(path)

        self._j = self._j + 1

        return path


    def _get_path_highFidelityModel(self):
        """
        Get the path of the high fidelity model for the current evaluation.
        @author: Markus Pfeil
        """
        path = os.path.join(self._path, SBO_Constants.PATH_ITERATION.format(self._k), SBO_Constants.PATH_HIGH_FIDELITY_MODEL)
        assert not os.path.exists(path)

        return path


    def _stepSizeNorm(self):
        """
        Calculate the absolute step size measured in the Euclidean norm
        @author: Markus Pfeil
        """
        return float(np.sum((np.array(self._modelParameter[self._k]) - np.array(self._modelParameter[self._k - 1]))**2))


    def _terminationConditionSBO(self):
        """
        Termination condition for the surrogate based optimization
        @author: Markus Pfeil
        """
        norm = self._stepSizeNorm() <= self._terminationConditionGamma
        trustRegionRadius = self._trustRegionRadius <= self._terminationConditionDelta
        maxIteration = self._terminationConditionMaxIteration is not None and self._k >= self._terminationConditionMaxIteration
        logging.debug('***Evaluate termination condition***\nNorm: {}\ntrustRegionRadius: {}\nmaxIteration: {}\nOverall: {}'.format(norm, trustRegionRadius, maxIteration, norm or trustRegionRadius or maxIteration))
        return norm or trustRegionRadius or maxIteration


    def _updateTrustRegionRadius(self, rho):
        """
        Update the trust region radius
        @author: Markus Pfeil
        """
        assert type(rho) in [float, np.float64, np.int64]

        if rho < self._trustRegionRadiusDecreaseBound:
            self._decreaseTrustRegionRadius()
        elif rho > self._trustRegionRadiusIncreaseBound:
            self._increaseTrustRegionRadius()


    def _decreaseTrustRegionRadius(self):
        """
        Decrease the trust region radius
        @author: Markus Pfeil
        """
        self._trustRegionRadius = self._trustRegionRadius / self._trustRegionRadiusDecreaseConstant


    def _increaseTrustRegionRadius(self):
        """
        Increase the trust region radius
        @author: Markus Pfeil
        """
        self._trustRegionRadius = self._trustRegionRadius * self._trustRegionRadiusIncreaseConstant


    def _calculateTargetTracerConcentration(self):
        """
        Calculate the tracer concentration for the target model parameter
        @author: Markus Pfeil
        """
        logging.debug('***Calculate the tracer concentration of the target model parameter***')
        #Run metos3d
        metos3dSimulationPath = os.path.join(self._path, 'TargetTracer')
        os.makedirs(metos3dSimulationPath, exist_ok=False)
        model = Metos3d.Metos3d(self._model, self._targetTracerModelTimestep, self._targetModelParameter, metos3dSimulationPath, modelYears = self._targetTracerModelYears, queue = self._queue, cores = self._cores)
        
        if self._useTrajectoryNorm:
            model.setCalculateTrajectory()
            model.setTrajectoryParameter(trajectoryYear=self._targetTracerModelYears+1, trajectoryStep=1)
        
        model.run()
        
        #Read tracer concentration
        self._targetTracerConcentration = model.readTracer()


    def _removeTargetTracerTrajectory(self):
        """
        Remove the tracer of the trajectory
        @author: Markus Pfeil
        """
        metos3dSimulationPath = os.path.join(self._path, 'TargetTracer')
        if self._useTrajectoryNorm and os.path.exists(metos3dSimulationPath) and os.path.isdir(metos3dSimulationPath):
            model = Metos3d.Metos3d(self._model, self._targetTracerModelTimestep, self._targetModelParameter, metos3dSimulationPath, modelYears = self._targetTracerModelYears, queue = self._queue, cores = self._cores)
            model.setCalculateTrajectory()
            model.removeTracer()


    def plot(self, plots=['Costfunction', 'StepSizeNorm', 'ParameterConvergence', 'AnnualCycle', 'Surface'], orientation='gmd', fontsize=8):
        """
        Plot the results for the optimization
        @author: Markus Pfeil
        """
        assert type(plots) is list
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        sboPlot = SurrogateBasedOptimizationPlot(self._optimizationId, queue=self._queue, cores=self._cores, orientation=orientation, fontsize=fontsize)

        path = os.path.join(SBO_Constants.PATH_FIGURE, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId))
        os.makedirs(path, exist_ok=True)

        for plot in plots:
            if plot == 'Costfunction':
                sboPlot.plot_costfunction()
                sboPlot.set_subplot_adjust(left=0.1275, bottom=0.15, right=0.99, top=0.995)
                filename = os.path.join(path, SBO_Constants.PATTERN_FIGURE_COSTFUNCTION.format(self._model, self._optimizationId))
                sboPlot.savefig(filename)
            elif plot == 'StepSizeNorm':
                sboPlot.plot_stepsize_norm()
                sboPlot.set_subplot_adjust(left=0.2, bottom=0.15, right=0.99, top=0.995)
                filename = os.path.join(path, SBO_Constants.PATTERN_FIGURE_STEPSIZENORM.format(self._model, self._optimizationId))
                sboPlot.savefig(filename)
            elif plot == 'ParameterConvergence':
                for i in range(Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._model]):
                    sboPlot.plot_parameter_convergence(parameterIndex=i)
                    filename = os.path.join(path, SBO_Constants.PATTERN_FIGURE_PARAMETERCONVERGENCE.format(self._model, i+1, self._optimizationId))
                    sboPlot.savefig(filename)
                    sboPlot.close_fig()
                    sboPlot.reinitialize_fig(orientation=orientation, fontsize=fontsize)
            elif plot == 'AnnualCycle':
                latitude = 30.9375
                longitude = -120.9375
                depth = 0
                iterationList = [] #[0, 2, 5, 10]
                runMetos3d = False
                plotFigure = True
                sboPlot.plot_annual_cycles(iterationList, latitude=latitude, longitude=longitude, depth=depth, runMetos3d=runMetos3d, plot=plotFigure, remove=False) 
                sboPlot.set_subplot_adjust(left=0.155, bottom=0.16, right=0.995, top=0.995)
                filename = os.path.join(path, SBO_Constants.PATTERN_FIGURE_ANNUALCYCLE.format(self._model, latitude, longitude, depth, self._optimizationId))
                if plotFigure:
                    sboPlot.savefig(filename)
            elif plot == 'AnnualCycleParameter':
                latitude = 30.9375 #30.9375 #0.0 #30.9375
                longitude = -30.9375 #-30.9375 #90.0 #-120.9375
                depth = 4
                parameterIdList = [0, 1152, 1153] #[1, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173]  #[0, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163] #[0, 1152, 1153]
                labelList = None #TODO
                runMetos3d = False
                plotFigure = True
                sboPlot.plot_annual_cycles_parameter(parameterIdList=parameterIdList, labelList=labelList, plotSurrogate=True, latitude=latitude, longitude=longitude, depth=depth, runMetos3d=runMetos3d, plot=plotFigure, remove=False) 
                sboPlot.set_subplot_adjust(left=0.155, bottom=0.16, right=0.995, top=0.995)
                filename = os.path.join(path, SBO_Constants.PATTERN_FIGURE_ANNUALCYCLEPARAMETER.format(self._model, latitude, longitude, depth, self._optimizationId))
                if plotFigure:
                    sboPlot.savefig(filename)
            elif plot == 'Surface':
                sboPlot.plot_surface(tracerDifference=True, relativeError=True)
            elif plot == 'SurfaceParameter':
                parameterIdList = [0, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163]
                runMetos3d = False
                plotFigure = True
                tracerDifference = True
                relativeError = True
                sboPlot.plot_surface_parameter(parameterIdList=parameterIdList,  plotHighFidelityModel=not tracerDifference, tracerDifference=tracerDifference, relativeError=relativeError, runMetos3d=runMetos3d, plot=plotFigure)
            elif plot == 'SurfaceLowFidelityModel':
                parameterIdList = [0, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163]
                runMetos3d = False
                plotFigure = True
                relativeError = False
                sboPlot.plot_surface_lowFidelityModels(parameterIdList=parameterIdList, relativeError=relativeError, runMetos3d=runMetos3d, plot=plotFigure)
            else:
                logging.error('***No rule to plot {} ***'.format(plot))
                assert False

            sboPlot.close_fig()
            sboPlot.reinitialize_fig(orientation=orientation, fontsize=fontsize)
        sboPlot.close_connection()


    def backup(self, movetar=False):
        """
        Generate a backup of the SBO run using a tar file
        @author: Markus Pfeil
        """
        assert type(movetar) is bool

        act_path = os.getcwd()
        path = os.path.join(SBO_Constants.PATH, 'Optimization')
        path_SBO = os.path.join(path, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId))
        assert os.path.exists(path_SBO) and os.path.isdir(path_SBO)
        os.chdir(path_SBO)

        tarfilename = os.path.join(path, SBO_Constants.PATTERN_BACKUP_FILENAME.format(self._optimizationId, SBO_Constants.COMPRESSION))
        assert not os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'w:{}'.format(SBO_Constants.COMPRESSION), compresslevel=SBO_Constants.COMPRESSLEVEL)

        #Add the logfile to the backup
        logfile = os.path.join(path, 'Logfile', SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId))
        if not os.path.exists(SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)) and os.path.exists(logfile) and os.path.isfile(logfile):
            os.rename(logfile, os.path.join(path_SBO, SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)))

        if os.path.exists(SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)) and os.path.isfile(SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)):
            try:
                tar.add(SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId))
            except tarfile.TarError:
                logging.info('Can not add the logfile {} to the archiv'.format(SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)))
        else:
            logging.info('Logfile {} does not exist.'.format(SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)))

        #Add the joboutput to the backup
        joboutput = os.path.join(path, 'Logfile', SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId))
        if not os.path.exists(SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)) and os.path.exists(joboutput) and os.path.isfile(joboutput):
            os.rename(joboutput, os.path.join(path_SBO, SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)))

        if os.path.exists(SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)) and os.path.isfile(SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)):
            try:
                tar.add(SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId))
            except tarfile.TarError:
                logging.info('Can not add the joboutput {} to the archiv'.format(SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)))
        else:
            logging.info('Joboutput {} does not exist.'.format(SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)))

        #Add the target tracer to the backup
        pathTargetTracer = os.path.join('TargetTracer')
        if os.path.exists(pathTargetTracer) and os.path.isdir(pathTargetTracer):
            #Add metos3d joboutput
            pathTargetTracerJoboutput = os.path.join(pathTargetTracer, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
            if os.path.exists(pathTargetTracerJoboutput) and os.path.isfile(pathTargetTracerJoboutput):
                try:
                    tar.add(pathTargetTracerJoboutput)
                except:
                    logging.info('Can not add the Metos3d joboutput of the target tracer {}.'.format(pathTargetTracerJoboutput))
            else:
                logging.info('Metos3d joboutout of the target tracer {} does not exist.'.format(pathTargetTracerJoboutput))

            #Add the target tracer
            if os.path.exists(os.path.join(pathTargetTracer, 'Tracer')) and os.path.isdir(os.path.join(pathTargetTracer, 'Tracer')):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    for end in ['', '.info']:
                        tracerFilename = os.path.join(pathTargetTracer, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                        if os.path.exists(tracerFilename) and os.path.isfile(tracerFilename):
                            try:
                                tar.add(tracerFilename)
                            except:
                                logging.info('Can not add the target tracer {} to the archiv.'.format(tracerFilename))
                        else:
                            logging.info('Target tracer {} does not exist.'.format(tracerFilename))
            else:
                logging.info('Target tracer path {} does not exist.'.format(os.path.join(pathTargetTracer, 'Tracer')))

        #Add each iteration of the SBO run
        for iteration in range(self._sboDB.get_count_iterations(self._optimizationId)):
            pathIteration = os.path.join(SBO_Constants.PATH_ITERATION.format(iteration))
            if os.path.exists(pathIteration) and os.path.isdir(pathIteration):
                for directory in os.listdir(pathIteration):
                    pathFidelityModel = os.path.join(pathIteration, directory)
                    if os.path.isdir(pathFidelityModel):
                        #Add metos3d joboutput
                        pathTracerJoboutput = os.path.join(pathFidelityModel, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                        if os.path.exists(pathTracerJoboutput) and os.path.isfile(pathTracerJoboutput):
                            try:
                                tar.add(pathTracerJoboutput)
                            except:
                                logging.info('Can not add the Metos3d joboutput of the tracer {}.'.format(pathTracerJoboutput))
                        else:
                            logging.info('Metos3d joboutout of the tracer {} does not exist.'.format(pathTracerJoboutput))

                        #Add tracer
                        if os.path.exists(os.path.join(pathFidelityModel, 'Tracer')) and os.path.isdir(os.path.join(pathFidelityModel, 'Tracer')):
                            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                                #Add the input tracer predicted by the ann
                                if self._lowFidelityModelUseAnn and directory.startswith(SBO_Constants.PATH_LOW_FIDELITY_MODEL.split('_', 1)[0]):
                                    tracerFilename = os.path.join(pathFidelityModel, 'Tracer', Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer))
                                    if os.path.exists(tracerFilename) and os.path.isfile(tracerFilename):
                                        try:
                                            tar.add(tracerFilename)
                                        except:
                                            logging.info('Can not add the tracer {} to the archiv.'.format(tracerFilename))
                                    else:
                                        logging.info('Tracer {} does not exist.'.format(tracerFilename))

                                #Add the calculated tracer concentrations
                                for end in ['', '.info']:
                                    tracerFilename = os.path.join(pathFidelityModel, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                                    if os.path.exists(tracerFilename) and os.path.isfile(tracerFilename):
                                        try:
                                            tar.add(tracerFilename)
                                        except:
                                            logging.info('Can not add the tracer {} to the archiv.'.format(tracerFilename))
                                    else:
                                        logging.info('Tracer {} does not exist.'.format(tracerFilename))
                        else:
                            logging.info('Tracer path {} does not exist.'.format(os.path.join(pathFidelityModel, 'Tracer')))

        tar.close()

        #Move tarfile to TAPE_CACHE
        if movetar:
            shutil.move(tarfilename, os.path.join(SBO_Constants.PATH_BACKUP, SBO_Constants.PATTERN_BACKUP_FILENAME.format(self._optimizationId, SBO_Constants.COMPRESSION)))
        os.chdir(act_path)
    

    def restore(self, movetar=False, restoreLogfile=True, restoreJoboutput=True, restoreTargetTracer=True, restoreTracer=True, restoreHighFidelityModel=True, restoreLowFidelityModel=True):
        """
        Restore the simulation data of the SBO run
        @author: Markus Pfeil
        """
        assert type(movetar) is bool
        assert type(restoreLogfile) is bool
        assert type(restoreJoboutput) is bool
        assert type(restoreTargetTracer) is bool
        assert type(restoreTracer) is bool
        assert type(restoreHighFidelityModel) is bool
        assert type(restoreLowFidelityModel) is bool

        act_path = os.getcwd()
        path = os.path.join(SBO_Constants.PATH, 'Optimization')
        assert os.path.exists(path) and os.path.isdir(path)
        pathSBO = os.path.join(path, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId))
        os.makedirs(pathSBO, exist_ok=True)
        os.chdir(pathSBO)

        tarfilename = os.path.join(path, SBO_Constants.PATTERN_BACKUP_FILENAME.format(self._optimizationId, SBO_Constants.COMPRESSION))
        if not os.path.exists(tarfilename) and movetar:
            shutil.copy2(os.path.join(SBO_Constants.PATH_BACKUP, SBO_Constants.PATTERN_BACKUP_FILENAME.format(self._optimizationId, SBO_Constants.COMPRESSION)), tarfilename)
        assert os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'r:{}'.format(SBO_Constants.COMPRESSION))

        #Restore the logfile
        if restoreLogfile:
            logfile = SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)
            try:
                info = tar.extract(logfile, path=pathSBO)
            except (tarfile.TarError, KeyError):
                logging.info('There is not the logfile {} in the archiv'.format(logfile))

        #Restore the joboutput
        if restoreJoboutput:
            joboutput = SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)
            try:
                info = tar.extract(joboutput, path=pathSBO)
            except (tarfile.TarError, KeyError):
                logging.info('There is not the joboutput {} in the archiv'.format(jobputput))

        #Restore the target tracer
        if restoreTargetTracer:
            pathTargetTracer = os.path.join('TargetTracer')
            os.makedirs(os.path.join(pathSBO, pathTargetTracer), exist_ok=True)            

            #Restore metos3d joboutput
            if restoreJoboutput:
                pathTargetTracerJoboutput = os.path.join(pathTargetTracer, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                try:
                    info = tar.extract(pathTargetTracerJoboutput, path=pathSBO)
                except (tarfile.TarError, KeyError):
                    logging.info('There is not the Metos3d joboutput {} of the target tracer in the archiv'.format(pathTargetTracerJoboutput))

            #Restore the target tracer
            os.makedirs(os.path.join(pathSBO, pathTargetTracer, 'Tracer'), exist_ok=True)
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                for end in ['', '.info']:
                    tracerFilename = os.path.join(pathTargetTracer, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                    try:
                        info = tar.extract(tracerFilename, path=pathSBO)
                    except (tarfile.TarError, KeyError):
                        logging.info('There is not the tracer {} of the target tracer in the archiv'.format(tracerFilename))

        #Restore each iteration of the SBO run
        if restoreTracer:
            for iteration in range(self._sboDB.get_count_iterations(self._optimizationId)):
                pathIteration = os.path.join(SBO_Constants.PATH_ITERATION.format(iteration))
                os.makedirs(os.path.join(pathSBO, pathIteration), exist_ok=True)

                #Restore high fidelity model
                if restoreHighFidelityModel:
                    pathHighFidelityModel = os.path.join(pathIteration, SBO_Constants.PATH_HIGH_FIDELITY_MODEL)
                    os.makedirs(os.path.join(pathSBO, pathHighFidelityModel), exist_ok=True)
                    
                    #Restore metos3d joboutput
                    if restoreJoboutput:
                        pathTracerJoboutput = os.path.join(pathHighFidelityModel, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                        try:
                            info = tar.extract(pathTracerJoboutput, path=pathSBO)
                        except (tarfile.TarError, KeyError):
                            logging.info('There is not the Metos3d joboutput {} in the archiv'.format(pathTracerJoboutput))

                    #Restore the tracer
                    os.makedirs(os.path.join(pathSBO, pathHighFidelityModel, 'Tracer'), exist_ok=True)
                    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                        for end in ['', '.info']:
                            tracerFilename = os.path.join(pathHighFidelityModel, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                            try:
                                info = tar.extract(tracerFilename, path=pathSBO)
                            except (tarfile.TarError, KeyError):
                                logging.info('There is not the tracer {} in the archiv'.format(tracerFilename))

                #Restore low fidelity models
                if restoreLowFidelityModel:
                    i = 0
                    while(True):
                        pathLowFidelityModel = os.path.join(pathIteration, SBO_Constants.PATH_LOW_FIDELITY_MODEL.format(i))
                        
                        try:
                            info = tar.getmember(os.path.join(pathLowFidelityModel, Metos3d_Constants.PATTERN_OUTPUT_FILENAME))
                        except KeyError:
                            logging.info('There is not low fidelity directory {} in the archiv'.format(pathLowFidelityModel))
                            break
                        else:
                            os.makedirs(os.path.join(pathSBO, pathLowFidelityModel), exist_ok=True) 

                            #Restore metos3d joboutput
                            if restoreJoboutput:
                                pathTracerJoboutput = os.path.join(pathLowFidelityModel, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                                try:
                                    info = tar.extract(pathTracerJoboutput)
                                except (tarfile.TarError, KeyError):
                                    logging.info('There is not the Metos3d joboutput {} in the archiv'.format(pathTracerJoboutput))

                            #Restore the tracer
                            os.makedirs(os.path.join(pathSBO, pathLowFidelityModel, 'Tracer'), exist_ok=True)
                            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                                #Restore the input tracer predicted by the ann
                                if self._lowFidelityModelUseAnn:
                                    tracerFilename = os.path.join(pathLowFidelityModel, 'Tracer', Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer))
                                    try:
                                        info = tar.extract(tracerFilename)
                                    except (tarfile.TarError, KeyError):
                                        logging.info('There is not the tracer {} in the archiv'.format(tracerFilename))

                                #Restore calculated tracer concentration
                                for end in ['', '.info']:
                                    tracerFilename = os.path.join(pathLowFidelityModel, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                                    try:
                                        info = tar.extract(tracerFilename)
                                    except (tarfile.TarError, KeyError):
                                        logging.info('There is not the tracer {} in the archiv'.format(tracerFilename))
                            i = i + 1

        tar.close()


    def remove(self, movetar=False):
        """
        Remove the simulation data of the SBO run
        @author: Markus Pfeil
        """
        assert type(movetar) is bool

        path = os.path.join(SBO_Constants.PATH, 'Optimization')
        pathSBO = os.path.join(path, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId))
        assert os.path.exists(pathSBO) and os.path.isdir(pathSBO)

        tarfilename = os.path.join(path, SBO_Constants.PATTERN_BACKUP_FILENAME.format(self._optimizationId, SBO_Constants.COMPRESSION))
        if not os.path.exists(tarfilename) and movetar:
            shutil.copy2(os.path.join(SBO_Constants.PATH_BACKUP, SBO_Constants.PATTERN_BACKUP_FILENAME.format(self._optimizationId, SBO_Constants.COMPRESSION)), tarfilename)
        assert os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'r:{}'.format(SBO_Constants.COMPRESSION))

        #Remove the logfile
        logfile = SBO_Constants.PATTERN_LOGFILE.format(self._optimizationId)
        try:
            info = tar.getmember(logfile)
        except KeyError:
            logging.info('There is not the logfile {} in the archiv'.format(logfile))
        else:
            if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, logfile)) and os.path.isfile(os.path.join(pathSBO, logfile)):
                os.remove(os.path.join(pathSBO, logfile))

        #Remove the joboutput
        joboutput = SBO_Constants.PATTERN_JOBOUTPUT.format(self._optimizationId)
        try:
            info = tar.getmember(joboutput)
        except KeyError:
            logging.info('There is not the joboutput {} in the archiv'.format(joboutput))
        else:
            if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, joboutput)) and os.path.isfile(os.path.join(pathSBO, joboutput)):
                os.remove(os.path.join(pathSBO, joboutput))

        #Remove the target tracer to the backup
        pathTargetTracer = os.path.join('TargetTracer')
        if os.path.exists(os.path.join(pathSBO, pathTargetTracer)) and os.path.isdir(os.path.join(pathSBO, pathTargetTracer)):
            #Remove metos3d joboutput
            pathTargetTracerJoboutput = os.path.join(pathTargetTracer, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
            try:
                info = tar.getmember(pathTargetTracerJoboutput)
            except KeyError:
                logging.info('There is not the Metos3d joboutput {} of the target tracer in the archiv'.format(pathTargetTracerJoboutput))
            else:
                if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, pathTargetTracerJoboutput)) and os.path.isfile(os.path.join(pathSBO, pathTargetTracerJoboutput)):
                    os.remove(os.path.join(pathSBO, pathTargetTracerJoboutput))

            #Remove the target tracer
            if os.path.exists(os.path.join(pathSBO, pathTargetTracer, 'Tracer')) and os.path.isdir(os.path.join(pathSBO, pathTargetTracer, 'Tracer')):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    for end in ['', '.info']:
                        tracerFilename = os.path.join(pathTargetTracer, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                        try:
                            info = tar.getmember(tracerFilename)
                        except KeyError:
                            logging.info('There is not the tracer {} of the target tracer in the archiv'.format(tracerFilename))
                        else:
                            if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, tracerFilename)) and os.path.isfile(os.path.join(pathSBO, tracerFilename)):
                                os.remove(os.path.join(pathSBO, tracerFilename))
            
                #Remove tracer directory
                try:
                    os.rmdir(os.path.join(pathSBO, pathTargetTracer, 'Tracer'))                
                except OSError as ex:
                    if ex.errno == errno.ENOTEMPTY:
                        logging.info('Tracer directory {} of the target tracer is not empty.'.format(os.path.join(pathSBO, pathTargetTracer, 'Tracer')))
            else:
                logging.info('Target tracer path {} does not exist.'.format(os.path.join(pathTargetTracer, 'Tracer')))

            #Remove target tracer directory
            try:
                os.rmdir(os.path.join(pathSBO, pathTargetTracer))
            except OSError as ex:
                if ex.errno == errno.ENOTEMPTY:
                    logging.info('Target tracer directory {} is not empty.'.format(os.path.join(pathSBO, pathTargetTracer)))

        #Remove each iteration of the SBO run
        for iteration in range(self._sboDB.get_count_iterations(self._optimizationId)+1):
            pathIteration = os.path.join(SBO_Constants.PATH_ITERATION.format(iteration))
            if os.path.exists(os.path.join(pathSBO, pathIteration)) and os.path.isdir(os.path.join(pathSBO, pathIteration)):
                for directory in os.listdir(os.path.join(pathSBO, pathIteration)):
                    pathFidelityModel = os.path.join(pathIteration, directory)
                    if os.path.isdir(os.path.join(pathSBO, pathFidelityModel)):
                        #Remove metos3d joboutput
                        pathTracerJoboutput = os.path.join(pathFidelityModel, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                        try:
                            info = tar.getmember(pathTracerJoboutput)
                        except KeyError:
                            logging.info('There is not the Metos3d joboutput {} in the archiv'.format(pathTracerJoboutput))
                        else:
                            if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, pathTracerJoboutput)) and os.path.isfile(os.path.join(pathSBO, pathTracerJoboutput)):
                                os.remove(os.path.join(pathSBO, pathTracerJoboutput))

                        #Remove the tracer
                        if os.path.exists(os.path.join(pathSBO, pathFidelityModel, 'Tracer')) and os.path.isdir(os.path.join(pathSBO, pathFidelityModel, 'Tracer')):
                            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                                #Restore the input tracer predicted by the ann
                                if self._lowFidelityModelUseAnn and directory.startswith(SBO_Constants.PATH_LOW_FIDELITY_MODEL.split('_', 1)[0]):
                                    tracerFilename = os.path.join(pathFidelityModel, 'Tracer', Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer))
                                    try:
                                        info = tar.getmember(tracerFilename)
                                    except KeyError:
                                        logging.info('There is not the tracer {} in the archiv'.format(tracerFilename))
                                    else:
                                        if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, tracerFilename)) and os.path.isfile(os.path.join(pathSBO, tracerFilename)):
                                            os.remove(os.path.join(pathSBO, tracerFilename))

                                for end in ['', '.info']:
                                    tracerFilename = os.path.join(pathFidelityModel, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer), end))
                                    try:
                                        info = tar.getmember(tracerFilename)
                                    except KeyError:
                                        logging.info('There is not the tracer {} in the archiv'.format(tracerFilename))
                                    else:
                                        if info.isfile() and info.size > 0 and os.path.exists(os.path.join(pathSBO, tracerFilename)) and os.path.isfile(os.path.join(pathSBO, tracerFilename)):
                                            os.remove(os.path.join(pathSBO, tracerFilename))

                            #Remove the tracer directory
                            try:
                                os.rmdir(os.path.join(pathSBO, pathFidelityModel, 'Tracer'))
                            except OSError as ex:
                                if ex.errno == errno.ENOTEMPTY:
                                    logging.info('Tracer directory {} is not empty.'.format(os.path.join(pathSBO, pathFidelityModel, 'Tracer')))
                        else:
                            logging.info('Tracer path {} does not exist.'.format(os.path.join(pathFidelityModel, 'Tracer')))

                        #Remove the fidelity model directory
                        try:
                            os.rmdir(os.path.join(pathSBO, pathFidelityModel))
                        except OSError as ex:
                            if ex.errno == errno.ENOTEMPTY:
                                logging.info('Fidelity model directory {} is not empty.'.format(os.path.join(pathSBO, pathFidelityModel)))

            #Remove iteration directory
            try:
                os.rmdir(os.path.join(pathSBO, pathIteration))
            except OSError as ex:
                if ex.errno == errno.ENOTEMPTY:
                    logging.info('Iteration directory {} is not empty.'.format(os.path.join(pathSBO, pathIteration)))

        #Remove directory of the SBO run
        try:
            os.rmdir(pathSBO)
        except OSError as ex:
            if ex.errno == errno.ENOTEMPTY:
                logging.info('SBO run directory {} is not empty.'.format(pathSBO))

        tar.close()

