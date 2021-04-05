#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os
import matplotlib.pyplot as plt

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.metos3d.Metos3d import readBoxVolumes
import metos3dutil.petsc.petscfile as petsc
from metos3dutil.plot.surfaceplot import SurfacePlot
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsPlot import TimestepsPlot


def main(orientation='gmd', fontsize=8, plotSpinup=False, plotNorm=False, plotRelationSpinupNorm=False, plotOscillationParameter=False, plotRelationErrorReduction=False, plotSurface=True):
    """
    Plot the results using different time steps for the spin up

    Create plots of the results using different time steps for the spin up
    calculation.

    Parameters
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    plotSpinup : bool, default: False
        If True, plot the figure of the spin up norm
    plotNorm : bool, default: False
        If True, plot the figures for the norm
    plotRelationSpinupNorm : bool, default: False
        If True, plot the figures of the relation between spin up norm and
        relative error
    plotOscillationParameter : bool, default: False
        If True, plot the radar chart of the model parameter for oscillating
        spin up norms
    """
    assert type(orientation) is str
    assert type(fontsize) is int and 0 < fontsize
    assert type(plotSpinup) is bool
    assert type(plotNorm) is bool

    timestepPlot = TimestepsPlots(orientation=orientation, fontsize=fontsize)

    #Spin up norm plot
    if plotSpinup:
        parameterId = 0
        kwargs = {'N': {}, 'N-DOP': {}, 'NP-DOP': {}, 'MITgcm-PO4-DOP': {}}
        kwargs['NPZ-DOP'] = {'timestepListSubPlot': [1, 2, 4, 8], 'axesResultSmall': [.61, .35, .3, .25], 'subplot_adjust': {'left': 0.1525, 'bottom': 0.165, 'right': 0.962, 'top': 0.84}, 'legend_box': True}
        kwargs['NPZD-DOP'] = kwargs['NPZ-DOP']
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            timestepPlot.plotSpinupData(metos3dModel, parameterId, **kwargs[metos3dModel])


    #Plot of the norm data
    if plotNorm:
        year = 10000
        for norm in ['2', 'BoxweightedVol']:
            for parameterId in range(0, 1): #Timesteps_Constants.PARAMETERID_MAX+1):
                for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                    timestepPlot.plotNormData(metos3dModel, parameterId, norm=norm,  year=year)


    #Plot relation between spin up norm and relative error
    if plotRelationSpinupNorm:
        year = 10000
        kwargs = {}
        kwargs['N'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}, 'legend_box': True}
        kwargs['N-DOP'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.9825, 'top': 0.9025}, 'legend_box': True}
        kwargs['NP-DOP'] = kwargs['N']
        kwargs['NPZ-DOP'] = kwargs['N']
        kwargs['NPZD-DOP'] = kwargs['N']
        kwargs['MITgcm-PO4-DOP'] = kwargs['N']
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            timestepPlot.plotScatterSpinupNorm(metos3dModel, year=year, **kwargs[metos3dModel])


    #Plot relation between spin up norm and relative error reduction
    if plotRelationErrorReduction:
        year = 10000
        kwargs = {}
        kwargs['N'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.995, 'top': 0.9025}, 'legend_box': True}
        kwargs['N-DOP'] = {'subplot_adjust': {'left': 0.145, 'bottom': 0.17, 'right': 0.9825, 'top': 0.9025}, 'legend_box': True}
        kwargs['NP-DOP'] = kwargs['N']
        kwargs['NPZ-DOP'] = kwargs['N']
        kwargs['NPZD-DOP'] = kwargs['N']
        kwargs['MITgcm-PO4-DOP'] = kwargs['N']
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            timestepPlot.plotScatterErrorReduction(metos3dModel, year=year, **kwargs[metos3dModel])


    #Plot model parameter of oscillating spin up norms
    if plotOscillationParameter:
        for metos3dModel in ['NP-DOP', 'NPZ-DOP', 'NPZD-DOP']:
            for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
                timestepPlot.plotOscillationParameter(metos3dModel, timestep)

    #Tracer concentration plot
    if plotSurface:
        parameterId = 0
        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
                timestepPlot.plotTracerConcentrationSurface(metos3dModel, parameterId, timestep, plotSlice=True, slicenum=[117])



    timestepPlot.closeDatabaseConnection()



class TimestepsPlots():
    """
    Preparation of plots for the results using different time steps

    Preparation of plots for the results using different time steps for the
    spin up calculation.

    Attributes
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    """

    def __init__(self, orientation='gmd', fontsize=8):
        """
        Constructs the environment to plot the data using different time steps
        for the spin up calculation.

        Parameter
        ----------
        orientation : str, default: gmd
            Orientation of the figure
        fontsize : int, default: 8
            Fontsize used in the figure
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize

        self._orientation = orientation
        self._fontsize = fontsize

        self.__timestepPlot = TimestepsPlot(orientation=self._orientation, fontsize=self._fontsize)


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self.__timestepPlot.closeDatabaseConnection()


    def plotSpinupData(self, model, parameterId, ncol=3, subPlot=True, **kwargs):
        """
        Plot the spin up norm.

        Plot the spin up norm for the given biogeochemical model and
        parameterId. The plot includes the spin up norm for the different time
        steps.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: True
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        **kwargs : dict
            Additional keyword arguments with keys:

            timestepList : list [int]
                Representation of the spin up norm for each specified timestep
            timestepListSubPlot : list [int]
                Representation of the spin up norm in the subplot for each
                specified timestep
            axesResultSmall : list [float]
                Dimensions of the subplot
            subplot_adjust : dict [str, float]
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert type(ncol) is int and 0 < ncol
        assert type(subPlot) is bool

        #Parse keyword arguments
        timestepList = kwargs['timestepList'] if 'timestepList' in kwargs and type(kwargs['timestepList']) is list else Metos3d_Constants.METOS3D_TIMESTEPS
        timestepListSubPlot = kwargs['timestepListSubPlot'] if 'timestepListSubPlot' in kwargs and type(kwargs['timestepListSubPlot']) is list else Metos3d_Constants.METOS3D_TIMESTEPS[:-2]
        axesResultSmall = kwargs['axesResultSmall'] if 'axesResultSmall' in kwargs and type(kwargs['axesResultSmall']) is list and len(kwargs['axesResultSmall']) == 4 else [.61, .30, .3, .34]
        subPlotModelYear = kwargs['subPlotModelYear'] if 'subPlotModelYear' in kwargs and type(kwargs['subPlotModelYear']) is int and 0 < kwargs['subPlotModelYear'] else 8000
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.1525, 'bottom': 0.165, 'right': 0.962, 'top': 0.995}

        self.__timestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__timestepPlot.plot_spinup_data(model, parameterId, ncol=ncol, subPlot=subPlot, timestepList=timestepList, timestepListSubPlot=timestepListSubPlot, axesResultSmall=axesResultSmall, subPlotModelYear=subPlotModelYear)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__timestepPlot.set_legend_box(ncol=ncol)

        self.__timestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        filenameSpinup = os.path.join(Timesteps_Constants.PATH_FIGURE, 'Spinup', model, Timesteps_Constants.PATTERN_FIGURE_SPINUP.format(model, parameterId))
        self.__timestepPlot.savefig(filenameSpinup)
        self.__timestepPlot.close_fig()


    def plotNormData(self, model, parameterId, norm='2', trajectory='', year=None, ncol=3, **kwargs):
        """
        Plot the relative error

        Plot the relative error of the spin up for the given biogeochemical
        model and parameterId. The plot includes the relative error for 
        different time steps. The tracer concentrations calculated with the
        spin up using the time step 1dt are used as reference solution.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        year : int or None, default: None
            Model year used for the reference solution
            If None, use the same year for the reference solution,
            respectively
        ncol : int, default: 3
            Number of columns for the legend
        **kwargs : dict
            Additional keyword arguments with keys:

            timestepList : list [int]
                Representation of the spin up norm for each specified timestep
            subplot_adjust : dict [str, float]
                Adjustment of the subplot using the keys left, bottom, right
                and top
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert trajectory in ['', 'trajectory']
        assert norm in DB_Constants.NORM
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol

        #Parse keyword arguments
        timestepList = kwargs['timestepList'] if 'timestepList' in kwargs and type(kwargs['timestepList']) is list else Metos3d_Constants.METOS3D_TIMESTEPS
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.165, 'right': 0.962, 'top': 0.995}

        self.__timestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__timestepPlot.plot_tracer_norm_data(parameterId, model, norm=norm, trajectory=trajectory, year=year, ncol=ncol)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__timestepPlot.set_legend_box(ncol=ncol)

        self.__timestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        filenameNorm = os.path.join(Timesteps_Constants.PATH_FIGURE, 'Norm', Timesteps_Constants.PATTERN_FIGURE_NORM.format(trajectory, norm, model, parameterId))
        self.__timestepPlot.savefig(filenameNorm)
        self.__timestepPlot.close_fig()


    def plotScatterSpinupNorm(self, model, year=10000, norm='2', trajectory='', **kwargs):
        """
        Plot the spin up norm against the norm

        Plot the spin up norm value against the norm value for the given model
        year using the different time steps

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        **kwargs : dict
            Additional keyword arguments with keys:

            timestepList : list [int]
                Representation of the spin up norm for each specified timestep
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']

        #Parse keyword arguments
        timestepList = kwargs['timestepList'] if 'timestepList' in kwargs and type(kwargs['timestepList']) is list else Metos3d_Constants.METOS3D_TIMESTEPS[1:]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.145, 'right': 0.995, 'top': 0.995}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 6
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4

        self.__timestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__timestepPlot.plot_scatter_spinup_norm(model, year=year, norm=norm, trajectory=trajectory, timestepList=timestepList)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__timestepPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__timestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        filename = os.path.join(Timesteps_Constants.PATH_FIGURE, 'Norm', Timesteps_Constants.PATTERN_FIGURE_SPINUP_NORM.format(trajectory, norm, model))
        self.__timestepPlot.savefig(filename)
        self.__timestepPlot.close_fig()


    def plotScatterErrorReduction(self, model, year=10000, norm='2', **kwargs):
        """
        Plot the spin up norm against the error reduction

        Plot the spin up norm value against the reduction of the relative
        error in the given norm between the intial tracer concentration and the
        tracer concentration for the given model year using the norm for all
        model parameter and time steps

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : str, default: 2
            Descriptive string for the norm to be used
            (see util.metos3dutil.database.constants.NORM)
        **kwargs : dict
            Additional keyword arguments with keys:

            timestepList : list [int]
                Representation of the spin up norm for each specified timestep
            legend_box : bool
                If the value is True, plot the legend of the plot using an bbox
                above the plot
            ncol : int, default: 3
                Number of columns for the legend

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM

        #Parse keyword arguments
        timestepList = kwargs['timestepList'] if 'timestepList' in kwargs and type(kwargs['timestepList']) is list else Metos3d_Constants.METOS3D_TIMESTEPS[1:]
        subplot_adjust = kwargs['subplot_adjust'] if 'subplot_adjust' in kwargs and type(kwargs['subplot_adjust']) is dict and 'left' in kwargs['subplot_adjust'] and 'bottom' in kwargs['subplot_adjust'] and 'right' in kwargs['subplot_adjust'] and 'top' in kwargs['subplot_adjust'] else {'left': 0.145, 'bottom': 0.145, 'right': 0.995, 'top': 0.995}
        ncol = kwargs['ncol'] if 'ncol' in kwargs and type(kwargs['ncol']) is int and kwargs['ncol'] > 0 else 6
        handlelength = kwargs['handlelength'] if 'handlelength' in kwargs and type(kwargs['handlelength']) is float and 0.0 < kwargs['handlelength'] else 0.5
        handletextpad = kwargs['handlepadtext'] if 'handlepadtext' in kwargs and type(kwargs['handlepadtext']) is float and 0.0 < kwargs['handlepadtext'] else 0.4

        self.__timestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__timestepPlot.plot_scatter_error_reduction(model, year=year, norm=norm, timestepList=timestepList)

        if 'legend_box' in kwargs and type(kwargs['legend_box']) is bool and kwargs['legend_box']:
            self.__timestepPlot.set_legend_box(ncol=ncol, handlelength=handlelength, handletextpad=handletextpad)

        self.__timestepPlot.set_subplot_adjust(left=subplot_adjust['left'], bottom=subplot_adjust['bottom'], right=subplot_adjust['right'], top=subplot_adjust['top'])

        filename = os.path.join(Timesteps_Constants.PATH_FIGURE, 'Norm', Timesteps_Constants.PATTERN_FIGURE_ERROR_REDUCTION.format(norm, model))
        self.__timestepPlot.savefig(filename)
        self.__timestepPlot.close_fig()



    def plotOscillationParameter(self, model, timestep):
        """
        Scatter plot of the model parameter for oscillating spin up norms

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : int
            Timestep used for the spin up calculation

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        self.__timestepPlot._init_plot(orientation=self._orientation, fontsize=self._fontsize)
        self.__timestepPlot.plot_oscillation_model_parameter(model, timestep)
        self.__timestepPlot.set_subplot_adjust(left=0.07, bottom=0.1, right=0.995, top=0.99)

        filename = os.path.join(Timesteps_Constants.PATH_FIGURE, Timesteps_Constants.PATTERN_FIGURE_OSCILLATION_PARAMETER.format(model, timestep))
        self.__timestepPlot.savefig(filename)
        self.__timestepPlot.close_fig()


    def plotTracerConcentrationSurface(self, model, parameterId, timestep, tracerDifference=True, relativeError=True, cmap=None, plotSurface=True, plotSlice=False, slicenum=None, orientation='etnasp'):
        """
        Plot the tracer concentration for the given layer

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        timestep : int
            Timestep used for the spin up calculation
        tracerDifference : bool, default: True
            If True, plot the tracer concentration difference between the
            concentration calculated with the given time step and the
            concentration calculated with the time step 1dt
        relativeError : bool, default: True
            If True
        cmap :

        plotSurface : bool, default: True
            If True, plot the tracer concentration at the surface
        plotSlice : bool, default: True
            If True, plot the slices of the tracer concentration
        slicenum : None or list [int], default: None
            The list slice of the tracer concentration for the given boxes.

        NOTES
        -----
        The figure is saved in the directory defined in
        timesteps.constants.PATH_FIGURE.
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tracerDifference) is bool
        assert type(relativeError) is bool
        assert type(plotSurface) is bool
        assert type(plotSlice) is bool
        assert slicenum is None or isinstance(slicenum, list)

        filenameTracer = os.path.join(Timesteps_Constants.PATH, 'Timesteps', model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(timestep), 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT)

        #Check if the tracer exists
        tracerExists = True
        for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
            tracerExists = tracerExists and os.path.exists(filenameTracer.format(tracer)) and os.path.isfile(filenameTracer.format(tracer))

        if tracerExists:
            #Read tracer concentration
            tracerConcentration = np.zeros(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[model])))
            i = 0
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
                tracerConcentration[:,i] = petsc.readPetscFile(filenameTracer.format(tracer))
                i += 1
        else:
            #Missing tracer file
            assert False

        #Calculate the norm of the tracer concentration vector
        if relativeError:
            filenameTracer1dt = os.path.join(Timesteps_Constants.PATH, 'Timesteps', model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(1), 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT)
            tracer1dt = np.zeros(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[model])))
            i = 0
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
                tracer1dt[:,i] = petsc.readPetscFile(filenameTracer1dt.format(tracer))
                i += 1
            normValue = np.linalg.norm(tracer1dt)
        else:
            normValue = 1.0

        #Plot the tracer concentration for every tracer
        i = 0
        for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
            if tracerDifference:
                v1d = np.divide(np.fabs(tracerConcentration[:,i] - tracer1dt[:,i]), normValue)
            else:
                v1d = np.divide(np.fabs(tracerConcentration[:,i]), normValue)

            surface = SurfacePlot(orientation=orientation)
            if plotSlice:
                surface.init_subplot(1, 2, orientation=orientation, gridspec_kw={'width_ratios': [9,5]})

            #Plot the surface concentration
            meridians = None if slicenum is None else [np.mod(Metos3d_Constants.METOS3D_GRID_LONGITUDE * x, 360) for x in slicenum]
            cntr = surface.plot_surface(v1d, projection='robin', levels=50, ticks=plt.LinearLocator(6), format='%.1e', pad=0.05, extend='max', meridians=meridians, colorbar=False)

            surface.set_subplot(0,1)

            #Plot the slice plan of the concentration
            if plotSlice and slicenum is not None:
                for s in slicenum:
                    surface.plot_slice(v1d, s, levels=50, ticks=plt.LinearLocator(6), format='%.1e', pad=0.02, extend='max', colorbar=False)

            plt.tight_layout(pad=0.05, w_pad=0.15)
            cbar = surface._fig.colorbar(cntr, ax=surface._axes[0], format='%.1e', ticks=plt.LinearLocator(5), pad=0.02, aspect=40, extend='max', orientation='horizontal', shrink=0.8)

            filenameSurface = os.path.join(Timesteps_Constants.PATH_FIGURE, 'Surface', model, Timesteps_Constants.PATTERN_FIGURE_SURFACE.format(model, timestep, parameterId, tracer, relativeError, tracerDifference))
            surface.savefig(filenameSurface)
            plt.close('all')
            i = i + 1




if __name__ == '__main__':
    main()

