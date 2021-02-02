#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os
import shutil

from metos3dutil.plot.plot import Plot
from metos3dutil.plot.surface import plotSurface
import metos3dutil.metos3d.Metos3d as Metos3d
import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants
import metos3dutil.petsc.petscfile as petsc
from sbo.AbstractClassSurrogateBasedOptimization import AbstractClassSurrogateBasedOptimization
import sbo.constants as SBO_Constants


class SurrogateBasedOptimizationPlot(Plot, AbstractClassSurrogateBasedOptimization):

    def __init__(self, optimizationId, nodes=NeshCluster_Constants.DEFAULT_NODES, orientation='lc1', fontsize=8, dbpath=SBO_Constants.DB_PATH, cmap=None):
        """
        Initialize the plot
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId
        assert type(nodes) is int and 0 < nodes
        assert type(fontsize) is int and 0 < fontsize
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)

        Plot.__init__(self, cmap=cmap, orientation=orientation, fontsize=fontsize)
        AbstractClassSurrogateBasedOptimization.__init__(self, optimizationId, nodes=nodes, dbpath=dbpath)

        self._path = os.path.join(SBO_Constants.PATH, 'Optimization', SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId))
        self._pathPlot = os.path.join(self._path, 'Plot')


    def plot_costfunction(self, optimizationIdList=[], costfunctionModel='HighFidelityModel', acceptedOnly=True):
        """
        Plot the cost function value for the whole optimization
        @author: Markus Pfeil
        """
        assert type(optimizationIdList) is list
        assert costfunctionModel in ['HighFidelityModel', 'Optimization']
        assert type(acceptedOnly) is bool

        if len(optimizationIdList) == 0:
            optimizationIds = [self._optimizationId]
        else:
            optimizationIds = optimizationIdList

        for optimizationId in optimizationIds:
            assert type(optimizationId) is int and 0 <= optimizationId
  
            costfunctionValues = self._sboDB.get_costfunctionValue(optimizationId, costfunctionModel=costfunctionModel, acceptedOnly=acceptedOnly)

            if not acceptedOnly:
                for i in range(costfunctionValues.shape[1]):
                    costfunctionValues[0,i] = i

            try:
                self._axesResult.plot(costfunctionValues[0,:], costfunctionValues[1,:])
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error mesage: Did not plot the result")
        
        self._axesResult.set_yscale('log', basey=10) 
        self._axesResult.set_xlabel(r'Iteration')
        self._axesResult.set_ylabel(r'$J(y_f(\mathbf{u}_k))$')


    def plot_stepsize_norm(self, optimizationIdList=[]):
        """
        Plot the step size norm for the whole optimization
        @author: Markus Pfeil
        """
        assert type(optimizationIdList) is list

        if len(optimizationIdList) == 0:
            optimizationIds = [self._optimizationId]
        else:
            optimizationIds = optimizationIdList

        for optimizationId in optimizationIds:
            assert type(optimizationId) is int and 0 <= optimizationId

            model = self._sboDB.get_model(optimizationId)
            parameterIds = self._sboDB.get_iteration_parameterIds(optimizationId)
            parameterDic = {}
            data = np.zeros(shape=(2, len(parameterIds)-1))
            for (iteration, parameterId) in parameterIds:
                parameterDic[iteration] = self._sboDB.get_parameter(parameterId, model)
                if iteration > 0:
                    data[0, iteration-1] = iteration
                    data[1, iteration-1] = np.sqrt(np.sum((parameterDic[iteration] - parameterDic[iteration-1])**2)) / np.sqrt(np.sum(parameterDic[iteration-1])**2)

            try:
                self._axesResult.plot(data[0,:], data[1,:])
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error mesage: Did not plot the result")

        self._axesResult.set_yscale('log', basey=10) 
        self._axesResult.set_xlabel(r'Iteration')
        self._axesResult.set_ylabel(r'$\frac{\left\| \mathbf{u}_k - \mathbf{u}_{k-1} \right\|_2}{\left\| \mathbf{u}_{k-1} \right\|_2}$')


    def plot_parameter_convergence(self, optimizationIdList=[], parameterIndex=0):
        """
        Plot the convergence of the single parameter values for each iteration of the SBO run
        @author: Markus Pfeil
        """
        assert type(optimizationIdList) is list
        assert type(parameterIndex) is int and 0 <= parameterIndex

        if len(optimizationIdList) == 0:
            optimizationIds = [self._optimizationId]
        else:
            optimizationIds = optimizationIdList

        for optimizationId in optimizationIds:
            assert type(optimizationId) is int and 0 <= optimizationId
        
            model = self._sboDB.get_model(optimizationId)
            parameterIds = self._sboDB.get_iteration_parameterIds(optimizationId)
            
            data = np.zeros(shape=(2, len(parameterIds)))
            dataTargetParameter = np.zeros(shape=(2, len(parameterIds)))
            for (iteration, parameterId) in parameterIds:
                data[0, iteration] = iteration
                data[1, iteration] = self._sboDB.get_parameter(parameterId, model)[parameterIndex]
                dataTargetParameter[0, iteration] = iteration

            dataTargetParameter[1,:] = self._sboDB.get_target_parameter(optimizationId)[parameterIndex]

            try:
                self._axesResult.plot(data[0,:], data[1,:], label='$\mathbf{u}_k$')
                self._axesResult.plot(dataTargetParameter[0,:], dataTargetParameter[1,:], label='$\mathbf{u}_d$')
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error mesage: Did not plot the result")

        self._axesResult.set_xlabel(r'Iteration')
        unit = '' if Metos3d_Constants.PARAMETER_UNITS_LATEX[Metos3d_Constants.PARAMETER_RESTRICTION[model]][parameterIndex] == '' else '\medskip \left[ {} \\right]'.format(Metos3d_Constants.PARAMETER_UNITS_LATEX[Metos3d_Constants.PARAMETER_RESTRICTION[model]][parameterIndex])
        self._axesResult.set_ylabel(r'$\mathbf{{u}}_{{k,{:d}}} = {}{}$'.format(parameterIndex + 1, Metos3d_Constants.PARAMETER_NAMES_LATEX[Metos3d_Constants.PARAMETER_RESTRICTION[model]][parameterIndex], unit))
        self._axesResult.legend(loc='best')


    def plot_annual_cycles(self, iterationList=[], plotTargetTracer=True, plotHighFidelityModel=True, plotLowFidelityModel=False, tracerNum=0, latitude=0.0, longitude=0.0, depth=0, runMetos3d=False, plot=True, remove=False):
        """
        Plot the annual cycle of a tracer for the given box
        @author: Markus Pfeil
        """
        assert type(iterationList) is list
        assert type(plotTargetTracer) is bool
        assert type(plotHighFidelityModel) is bool
        assert type(plotLowFidelityModel) is bool
        assert type(tracerNum) is int and 0 <= tracerNum
        assert type(latitude) is float and -90.0 <= latitude and latitude <= 90.0
        assert type(longitude) is float and -180.0 <= longitude and longitude <= 180.0
        assert type(depth) is int and 0 <= depth and depth <= 15
        assert type(runMetos3d) is bool
        assert type(plot) is bool
        assert type(remove) is bool

        #Plot the annual cycle using the target parameter
        if plotTargetTracer:
            targetTracer = self._calculate_trajectory('TargetTracer', runMetos3d=runMetos3d, remove=remove)
            if plot and targetTracer is not None:
                v1dTargetTracer = self._select_trajactory_box(targetTracer, tracerNum=tracerNum, latitude=latitude, longitude=longitude, depth=depth)

                try:
                    self._axesResult.plot(v1dTargetTracer[0,:], v1dTargetTracer[1,:], label='$y_d$')
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error mesage: Did not plot the result")

        #Calculate and plot the annual cycle for the iteration in the SBO optimization run
        if len(iterationList) == 0: 
            iterations = range(self._sboDB.get_count_iterations(self._optimizationId))
        else:
            iterations = iterationList

        for iteration in iterations:
            #Read tracer trajectory of the high fidelity model
            if plotHighFidelityModel:
                tracerHighFidelityModel = self._calculate_trajectory('highFidelityModel', iteration=iteration, runMetos3d=runMetos3d, remove=remove)

                #Plot the annual cycle of the high fidelity model
                if plotHighFidelityModel and plot and tracerHighFidelityModel is not None:
                    v1dHighFidelityModel = self._select_trajactory_box(tracerHighFidelityModel, tracerNum=tracerNum, latitude=latitude, longitude=longitude, depth=depth)

                    try:
                        self._axesResult.plot(v1dHighFidelityModel[0,:], v1dHighFidelityModel[1,:], label='$y_f\left(\mathbf{{u}}_{{{:d}}}\\right)$'.format(iteration))
                    except IOError as e:
                        print("Error message: " + e.args[1])
                        print("Error mesage: Did not plot the result")

            #Read the tracer trajectory of the low fidelity model
            if plotLowFidelityModel:
                tracerLowFidelityModel = self._calculate_trajectory('lowFidelityModel', iteration=iteration, runMetos3d=runMetos3d, remove=remove)

                #Plot the annual cycle of the low fidelity model
                if plotLowFidelityModel and plot and tracerLowFidelityModel is not None:
                    v1dLowFidelityModel = self._select_trajactory_box(tracerLowFidelityModel, tracerNum=tracerNum, latitude=latitude, longitude=longitude, depth=depth)

                    try:
                        self._axesResult.plot(v1dLowFidelityModel[0,:], v1dLowFidelityModel[1,:], label='$y_c\left(\mathbf{{u}}_{{{:d}}}\\right)$'.format(iteration))
                    except IOError as e:
                        print("Error message: " + e.args[1])
                        print("Error mesage: Did not plot the result")

        self._axesResult.set_xlabel(r'Time steps')
        self._axesResult.set_ylabel(r'$N \left[ mmol \, P \, m^{-3} \right]$')
        self._axesResult.legend(loc='best')


    def create_nosiy_parameter(self, metos3dModel, parameterId=0, count=1, noise=0.1, decimals=5):
        """
        Create parameter to evaluate the surrogate.
        This functions creates a new parameter disturbing the parameter with the given parameterId using a random noise.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and 0 <= parameterId
        assert type(count) is int and 0 < count
        assert type(noise) is float and 0 < noise
        assert type(decimals) is int and 0 <= decimals

        parameter = self._sboDB.get_parameter(parameterId, metos3dModel)
        parameterIdList = []

        for _ in range(count):
            nosiyParameter = parameter + parameter * noise * np.random.normal(loc=0.0, scale=0.5, size=Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[metos3dModel])
            nosiyParameter = np.around(nosiyParameter, decimals=decimals)
            parameterIdList.append(self._sboDB.insert_parameter(list(nosiyParameter), metos3dModel))

        return parameterIdList


    def plot_annual_cycles_parameter(self, parameterIdList=[], labelList=None, plotHighFidelityModel=True, plotLowFidelityModel=True, plotSurrogate=False, tracerNum=0, latitude=0.0, longitude=0.0, depth=0, runMetos3d=False, plot=True, remove=False):
        """
        Plot the annual cycle of a tracer for the high fidelity model, low fidelity model and the evaluated surrogate.
        The surrogate is constructed for the first parameter.
        @author: Markus Pfeil
        """
        assert type(parameterIdList) is list
        assert labelList is None or type(labelList) is list and len(labelList) == len(parameterIdList)
        assert type(plotHighFidelityModel) is bool
        assert type(plotLowFidelityModel) is bool
        assert type(plotSurrogate) is bool
        assert type(tracerNum) is int and 0 <= tracerNum
        assert type(latitude) is float and -90.0 <= latitude and latitude <= 90.0
        assert type(longitude) is float and -180.0 <= longitude and longitude <= 180.0
        assert type(depth) is int and 0 <= depth and depth <= 15
        assert type(runMetos3d) is bool
        assert type(plot) is bool
        assert type(remove) is bool

        #Calculate and plot the annual cycle for the iteration in the SBO optimization run
        if len(parameterIdList) == 0:
            parameterIds = [parameterTuple[1] for parameterTuple in self._sboDB.get_iteration_parameterIds(self._optimizationId)]
        else:
            parameterIds = parameterIdList

        labelIndex = 0
        for parameterId in parameterIds:
            parameter = list(self._sboDB.get_parameter(parameterId, self._model))
            self._parameterPlotIndex = parameterId
            self._j = 0
            label = '\mathbf{{u}}_{{{:d}}}'.format(parameterId) if labelList is None else labelList[labelIndex]

            #Calculate the trajectory for the given parameter using the high fidelity model
            if plotHighFidelityModel or (plotSurrogate and labelIndex == 0):
                tracerHighFidelityModel = self._highFidelityModel(parameter, runMetos3d=runMetos3d, remove=remove)

                #Plot the annual cycle of the high fidelity model
                if plotHighFidelityModel and plot and tracerHighFidelityModel is not None:
                    v1dHighFidelityModel = self._select_trajactory_box(tracerHighFidelityModel, tracerNum=tracerNum, latitude=latitude, longitude=longitude, depth=depth)

                    try:
                        self._axesResult.plot(v1dHighFidelityModel[0,:], v1dHighFidelityModel[1,:], linestyle='solid', color=self._colors[labelIndex], label='$y_f\left({:s}\\right)$'.format(label))
                    except IOError as e:
                        print("Error message: " + e.args[1])
                        print("Error mesage: Did not plot the result")

            #Calculate the trajectory for the given parameter using the low fidelity model
            if plotLowFidelityModel or (plotSurrogate and labelIndex == 0):
                tracerLowFidelityModel = self._lowFidelityModel(parameter, runMetos3d=runMetos3d, remove=remove)

                #Plot the annual cycle of the low fidelity model
                if plotLowFidelityModel and plot and tracerLowFidelityModel is not None:
                    v1dLowFidelityModel = self._select_trajactory_box(tracerLowFidelityModel, tracerNum=tracerNum, latitude=latitude, longitude=longitude, depth=depth)

                    try:
                        self._axesResult.plot(v1dLowFidelityModel[0,:], v1dLowFidelityModel[1,:], linestyle=(0, (5, 10)), color=self._colors[labelIndex], label='$y_c\left({:s}\\right)$'.format(label))
                    except IOError as e:
                        print("Error message: " + e.args[1])
                        print("Error mesage: Did not plot the result")

            #Construct the surrogate
            if labelIndex == 0 and tracerHighFidelityModel is not None and tracerLowFidelityModel is not None:
                tracerHighFidelityModel = tracerHighFidelityModel[::self._lowFidelityModelTimestep//self._highFidelityModelTimestep]
                self._constructSurrogate(tracerHighFidelityModel, tracerLowFidelityModel)

            #Plot the annual cycle of the surrogate
            if labelIndex > 0 and plotSurrogate:
                tracerSurrogate = self._evaluateSurrogate(parameter, runMetos3d=runMetos3d, remove=remove)

                if plot:
                    v1dSurrogate = self._select_trajactory_box(tracerSurrogate, tracerNum=tracerNum, latitude=latitude, longitude=longitude, depth=depth)

                    try:
                        self._axesResult.plot(v1dSurrogate[0,:], v1dSurrogate[1,:], linestyle=(0, (3, 10, 1, 10)), color=self._colors[labelIndex], label='$s_0\left({:s}\\right)$'.format(label))
                    except IOError as e:
                        print("Error message: " + e.args[1])
                        print("Error mesage: Did not plot the result")

            labelIndex += 1

        if remove and os.path.exists(self._pathPlot) and os.path.isdir(self._pathPlot):
            shutil.rmtree(self._pathPlot)

        self._axesResult.set_xlabel(r'Time steps')
        self._axesResult.set_ylabel(r'$N \left[ mmol \, P \, m^{-3} \right]$')
        self._axesResult.legend(loc='best')


    def _get_path_highFidelityModel(self):
        """
        Get the path of the high fidelity model for the current evaluation.
        @author: Markus Pfeil
        """
        return os.path.join(self._pathPlot, 'Parameter_{:0>4d}'.format(self._parameterPlotIndex), SBO_Constants.PATH_HIGH_FIDELITY_MODEL)


    def _get_path_lowFidelityModel(self):
        """
        Get the path of the low fidelity model for the current evaluation.
        @author: Markus Pfeil
        """
        path = os.path.join(self._pathPlot, 'Parameter_{:0>4d}'.format(self._parameterPlotIndex), SBO_Constants.PATH_LOW_FIDELITY_MODEL.format(self._j))
        self._j = self._j + 1
        return path


    def _calculate_trajectory(self, fidelityModel, iteration=None, runMetos3d=False, remove=False):
        """
        Calculate the trajectory for the given model.
        @author: Markus Pfeil
        """
        assert fidelityModel in ['TargetTracer', 'highFidelityModel', 'lowFidelityModel']
        assert fidelityModel == 'TargetTracer' and iteration is None or fidelityModel in ['highFidelityModel', 'lowFidelityModel'] and type(iteration) is int and 0 <= iteration
        assert type(runMetos3d) is bool
        assert type(remove) is bool

        #Parameter to calculte the trajectories
        if fidelityModel == 'TargetTracer':
            timestep = 64 #TODO SBO_Constants.TARGET_TRACER_MODEL_TIMESTEP
            path = os.path.join(self._path, 'TargetTracer')
            parameter = list(self._sboDB.get_target_parameter(self._optimizationId))
        elif fidelityModel == 'highFidelityModel':
            (_, timestep) = self._sboDB.get_highFidelityModel(self._optimizationId)
            path = os.path.join(self._path, SBO_Constants.PATH_ITERATION.format(iteration), SBO_Constants.PATH_HIGH_FIDELITY_MODEL)
            parameter = list(self._sboDB.get_parameter_iteration(self._optimizationId, iteration))
        elif fidelityModel == 'lowFidelityModel':
            (_, timestep) = self._sboDB.get_lowFidelityModel(self._optimizationId)
            path = os.path.join(self._path, SBO_Constants.PATH_ITERATION.format(iteration), SBO_Constants.PATH_LOW_FIDELITY_MODEL.format(0))
            parameter = list(self._sboDB.get_parameter_iteration(self._optimizationId, iteration))
        else:
            assert False

        #Calculate and plot the annual cycle for the target parameter
        metos3dSimulationPath = os.path.join(path, 'Temp_Trajectory')
        if os.path.exists(path) and os.path.isdir(path):
            #Copy the tracer file to use as input for metos3d
            if runMetos3d:
                pathTracer = os.path.join(metos3dSimulationPath, 'Tracer')
                os.makedirs(pathTracer, exist_ok=False)
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    shutil.copy2(os.path.join(path, 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer)), os.path.join(pathTracer, Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer)))
        
            #Run metos3d to calculate the trajectory
            metos3dModel = Metos3d.Metos3d(self._model, timestep, parameter, metos3dSimulationPath, modelYears = 0, nodes = self._nodes)
            metos3dModel.setCalculateOnlyTrajectory()
            
            if runMetos3d:
                metos3dModel.run()
            
            #Select the trajectory for the given box
            tracer = metos3dModel.readTracer()
        
            #Remove the directory of the trajectory
            if remove:
                metos3dModel.removeTracer()
                shutil.rmtree(metos3dSimulationPath)
        else:
            tracer = None

        return tracer


    def _select_trajactory_box(self, tracer, tracerNum=0, latitude=0.0, longitude=0.0, depth=0):
        """
        Select the trajectory for the given box
        @author: Markus Pfeil
        """
        assert type(tracer) is np.ndarray
        assert type(tracerNum) is int and 0 <= tracerNum
        assert type(latitude) is float and -90.0 <= latitude and latitude <= 90.0
        assert type(longitude) is float and -180.0 <= longitude and longitude <= 180.0
        assert type(depth) is int and 0 <= depth and depth <= 15

        #Calculate the box numbers for the latitude and longitude
        ny = int(longitude // Metos3d_Constants.METOS3D_GRID_LONGITUDE) if longitude >= 0 else int((360 // Metos3d_Constants.METOS3D_GRID_LONGITUDE) - (abs(longitude) // Metos3d_Constants.METOS3D_GRID_LONGITUDE))
        nx = int((90 // Metos3d_Constants.METOS3D_GRID_LATITUDE) + (latitude // Metos3d_Constants.METOS3D_GRID_LATITUDE)) if latitude >= 0 else int((90 // Metos3d_Constants.METOS3D_GRID_LATITUDE) - (abs(latitude) // Metos3d_Constants.METOS3D_GRID_LATITUDE)) 

        (steps, _, _) = tracer.shape

        v1d = np.zeros(shape=(2, steps), dtype=float)
        for step in range(steps):
            v1d[0, step] = step

            v3d = self._reorganize_data(tracer[step, tracerNum, :])
            v1d[1, step] = v3d[0, nx, ny, depth]

        return v1d


    def plot_surface(self, depth=0, projection='robin', tracer='N', tracerDifference=False, relativeError=False):
        """
        Plot the tracer concentration for the target tracer concentation and the tracer concentration calculated with the high fidelity model for every iteration of the optimization.
        @author: Markus Pfeil
        """
        assert type(depth) is int and 0 <= depth and depth <= Metos3d_Constants.METOS3D_GRID_DEPTH
        assert projection in ['cyl', 'robin']
        assert tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]
        assert type(tracerDifference) is bool
        assert type(relativeError) is bool

        #Set the norm value using the target tracer concentration
        if relativeError:
            targetTracer = np.zeros(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
            i = 0
            for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                tracerFilename = os.path.join(self._path, 'TargetTracer', 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
                targetTracer[:,i] = petsc.readPetscFile(tracerFilename)
                i = i + 1
            normValue = np.linalg.norm(targetTracer)
        else:
            normValue = 1.0

        #Read the target tracer concentration to calculate the difference of tracers
        plotList = []
        if tracerDifference:
            tracerFilename = os.path.join(self._path, 'TargetTracer', 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
            targetTracer = petsc.readPetscFile(tracerFilename)
        else:
            plotList = [('TargetTracer', False)]

        #Set the iteration to plot the tracer concentration at the surface (or deeper layer)
        for iteration in range(self._sboDB.get_count_iterations(self._optimizationId)):
            plotList.append((iteration, True))

        vmin = None
        vmax = None

        for (simulation, flag) in plotList:
            tracerFilename = os.path.join(self._path, '{:s}'.format(os.path.join(SBO_Constants.PATH_ITERATION.format(simulation), SBO_Constants.PATH_HIGH_FIDELITY_MODEL) if flag else simulation), 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
            tracerV1d = petsc.readPetscFile(tracerFilename)
            if tracerDifference and relativeError:
                v1d = np.divide(np.fabs(targetTracer - tracerV1d), normValue)
            elif tracerDifference and not relativeError:
                v1d = targetTracer - tracerV1d
            else:
                v1d =  np.divide(tracerV1d, normValue)

            if vmin is None:
                vmin = float(np.min(v1d)) if tracerDifference and not relativeError else 0.0
            if vmax is None:
                vmax = float(np.max(v1d))

            filename = os.path.join(SBO_Constants.PATH_FIGURE, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId), SBO_Constants.PATTERN_FIGURE_SURFACE.format(self._optimizationId, projection, self._model, depth, tracer, SBO_Constants.PATH_ITERATION.format(simulation) if flag else simulation, tracerDifference, relativeError))
            plotSurface(v1d, filename, depth=depth, projection=projection, vmin=vmin, vmax=vmax)


    def plot_surface_parameter(self, parameterIdList=[], plotHighFidelityModel=True, plotLowFidelityModel=True, plotSurrogate=True, depth=0, projection='robin', tracer='N', tracerDifference=False, relativeError=False, runMetos3d=False, plot=False, remove=False, vmin=None, vmax=None):
        """
        Plot the tracer concentration for different parameter sets using the high fidelity model, the low fidelity model and the surrogate. 
        The surrogate is built with the first parameter set and evaluated with the other parameter sets.
        @author: Markus Pfeil
        """
        assert type(parameterIdList) is list
        assert type(plotHighFidelityModel) is bool
        assert type(plotLowFidelityModel) is bool
        assert type(plotSurrogate) is bool
        assert type(depth) is int and 0 <= depth and depth <= Metos3d_Constants.METOS3D_GRID_DEPTH
        assert projection in ['cyl', 'robin']
        assert tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]
        assert type(tracerDifference) is bool
        assert type(relativeError) is bool
        assert (not plotHighFidelityModel or (plotHighFidelityModel and not tracerDifference))
        assert type(runMetos3d) is bool
        assert type(remove) is bool
        assert vmin is None or type(vmin) is float
        assert vmax is None or type(vmax) is float

        #Plot the tracer concentration for the iteration in the SBO optimization run
        if len(parameterIdList) == 0:
            parameterIds = [parameterTuple[1] for parameterTuple in self._sboDB.get_iteration_parameterIds(self._optimizationId)]
        else:
            parameterIds = parameterIdList

        for parameterId in parameterIds:
            parameter = list(self._sboDB.get_parameter(parameterId, self._model))
            self._parameterPlotIndex = parameterId
            self._j = 0
            normValue = 1.0

            #Calculate the trajectory for the given parameter using the high fidelity model
            if plotHighFidelityModel or (plotSurrogate and parameterId == parameterIds[0]) or tracerDifference:
                tracerHighFidelityModel = self._highFidelityModel(parameter, runMetos3d=runMetos3d, remove=remove)

                #Set the norm value using the tracer concentration used to build the surrogate
                if relativeError:
                    normValue = np.linalg.norm(tracerHighFidelityModel)

                #Plot the tracer concentration of the high fidelity model
                if plotHighFidelityModel and plot and tracerHighFidelityModel is not None:
                    if self._useTrajectoryNorm:
                        v1dHighFidelityModel = tracerHighFidelityModel[0, Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]
                    else:
                        v1dHighFidelityModel = tracerHighFidelityModel[Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]

                    if vmin is None:
                        vmin = float(np.min(v1dHighFidelityModel)) if tracerDifference and not relativeError else 0.0
                    if vmax is None:
                        vmax = float(np.max(v1dHighFidelityModel))

                    filename = os.path.join(SBO_Constants.PATH_FIGURE, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId), SBO_Constants.PATTERN_FIGURE_SURFACE_PARAMETER.format(self._optimizationId, projection, self._model, depth, tracer, parameterId, 'HighFidelityModel', tracerDifference, relativeError))
                    plotSurface(v1dHighFidelityModel, filename, depth=depth, projection=projection, vmin=vmin, vmax=vmax)

            #Calculate the trajectory for the given parameter using the low fidelity model
            if plotLowFidelityModel or (plotSurrogate and parameterId == parameterIds[0]):
                tracerLowFidelityModel = self._lowFidelityModel(parameter, runMetos3d=runMetos3d, remove=remove)

                #Plot the tracer concentration of the low fidelity model
                if plotLowFidelityModel and plot and tracerLowFidelityModel is not None:
                    if tracerDifference and relativeError:
                        v1d = np.divide(np.fabs(tracerHighFidelityModel - tracerLowFidelityModel), normValue)
                    elif tracerDifference and not relativeError:
                        v1d = tracerHighFidelityModel - tracerLowFidelityModel
                    else:
                        v1d =  np.divide(tracerLowFidelityModel, normValue)

                    if self._useTrajectoryNorm:
                        v1dLowFidelityModel = v1d[0, Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]
                    else:
                        v1dLowFidelityModel = v1d[Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]

                    filename = os.path.join(SBO_Constants.PATH_FIGURE, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId), SBO_Constants.PATTERN_FIGURE_SURFACE_PARAMETER.format(self._optimizationId, projection, self._model, depth, tracer, parameterId, 'LowFidelityModel', tracerDifference, relativeError))
                    plotSurface(v1dLowFidelityModel, filename, depth=depth, projection=projection, vmin=vmin, vmax=vmax)

            #Construct the surrogate
            if  parameterId == parameterIds[0] and tracerHighFidelityModel is not None and tracerLowFidelityModel is not None:
                tracerHighFidelityModel = tracerHighFidelityModel[::self._lowFidelityModelTimestep//self._highFidelityModelTimestep]
                self._constructSurrogate(tracerHighFidelityModel, tracerLowFidelityModel)

            #Plot the tracer concentration of the surrogate
            if parameterId != parameterIds[0] and plotSurrogate:
                tracerSurrogate = self._evaluateSurrogate(parameter, runMetos3d=runMetos3d, remove=remove)

                if tracerDifference and relativeError:
                    v1d = np.divide(np.fabs(tracerHighFidelityModel - tracerSurrogate), normValue)
                elif tracerDifference and not relativeError:
                    v1d = tracerHighFidelityModel - tracerSurrogate
                else:
                    v1d =  np.divide(tracerSurrogate, normValue)

                if self._useTrajectoryNorm:
                    v1dSurrogate = v1d[0, Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]
                else:
                    v1dSurrogate = v1d[Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]

                filename = os.path.join(SBO_Constants.PATH_FIGURE, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId), SBO_Constants.PATTERN_FIGURE_SURFACE_PARAMETER.format(self._optimizationId, projection, self._model, depth, tracer, parameterId, 'Surrogate', tracerDifference, relativeError))
                plotSurface(v1dSurrogate, filename, depth=depth, projection=projection, vmin=vmin, vmax=vmax)


    def plot_surface_lowFidelityModels(self, parameterIdList=[], depth=0, projection='robin', tracer='N', relativeError=False, runMetos3d=False, plot=False, remove=False, vmin=None, vmax=None):
        """
        Plot the tracer concentration difference between two approximations using the low fidelity model. The first parameter set is used already as reference.
        @author: Markus Pfeil
        """
        assert type(parameterIdList) is list
        assert type(depth) is int and 0 <= depth and depth <= Metos3d_Constants.METOS3D_GRID_DEPTH
        assert projection in ['cyl', 'robin']
        assert tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]
        assert type(relativeError) is bool
        assert type(runMetos3d) is bool
        assert type(remove) is bool
        assert vmin is None or type(vmin) is float
        assert vmax is None or type(vmax) is float

        #Plot the tracer concentration for low fidelity models of the iterations in the SBO optimization run
        if len(parameterIdList) == 0:
            parameterIds = [parameterTuple[1] for parameterTuple in self._sboDB.get_iteration_parameterIds(self._optimizationId)]
        else:
            parameterIds = parameterIdList

        #Do not use the trajectory to receive directly the tracer concentration of the ann prediction.
        if self._useTrajectoryNorm:
            self._setTrajectoryNorm(False)

        #Read the tracer concentration for the first parameter set as reference tracer concentration
        if len(parameterIds) > 0:
            parameterIdReference = parameterIds.pop(0)
            parameter = list(self._sboDB.get_parameter(parameterIdReference, self._model))
            tracerReference = self._lowFidelityModel(parameter, runMetos3d=runMetos3d, remove=remove)

            if relativeError:
                normValue = np.linalg.norm(tracerReference)
            else:
                normValue = 1.0

        for parameterId in parameterIds:
            parameter = list(self._sboDB.get_parameter(parameterId, self._model))
            self._parameterPlotIndex = parameterId
            self._j = 0

            tracerLowFidelityModel = self._lowFidelityModel(parameter, runMetos3d=runMetos3d, remove=remove)
            
            #Plot the tracer concentration of the low fidelity model
            if plot and tracerLowFidelityModel is not None:
                if relativeError:
                    v1d = np.divide(np.fabs(tracerReference - tracerLowFidelityModel), normValue)
                else:
                    v1d = tracerReference - tracerLowFidelityModel

                v1dLowFidelityModel = v1d[Metos3d_Constants.METOS3D_MODEL_TRACER[self._model].index(tracer), :]

                filename = os.path.join(SBO_Constants.PATH_FIGURE, SBO_Constants.PATH_OPTIMIZATION.format(self._optimizationId), SBO_Constants.PATTERN_FIGURE_SURFACE_LOWFIDELITYMODEL.format(self._optimizationId, projection, self._model, depth, tracer, parameterIdReference, parameterId, relativeError))
                plotSurface(v1dLowFidelityModel, filename, depth=depth, projection=projection, vmin=vmin, vmax=vmax)

        #Reset the use of the trajectory
        self._setTrajectoryNorm(self._sboDB.get_trajectoryNorm(self._optimizationId))

