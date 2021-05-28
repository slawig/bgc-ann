#!/usr/bin/env python
# -*- coding: utf8 -*

import matplotlib.pyplot as plt
import numpy as np
import os

import metos3dutil.database.constants as DB_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.petsc.petscfile as petsc
from metos3dutil.plot.plot import Plot
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsDatabase import Timesteps_Database


class TimestepsPlot(Plot):
    """
    Creation of plots using different time steps

    Creation of plots for the spin up calculation using different time steps

    Attributes
    ----------
    colorsTimestep : dict [int, str]
        Assignment of timesteps to different colors used in the plot
    """

    def __init__(self, orientation='gmd', fontsize=8, dbpath=Timesteps_Constants.DB_PATH, cmap=None, completeTable=True):
        """
        Constructs the environment to plot the data using different time steps

        Parameter
        ----------
        orientation : str, default: gmd
            Orientation of the figure
        fontsize : int, default: 8
            Fontsize used in the figure
        dbpath : str, default: timesteps.constants.DB_PATH
            Path to the sqlite database
        cmap : matplotlib.colors.Colormap or None, default: None
            Colormap used in the surface plot to visualize tracer
            concentrations
        completeTable : bool, default: True
            If the value is True, use all columns (even columns with value
            None) in SELECT queries on the database
        """
        assert type(orientation) is str
        assert type(fontsize) is int and 0 < fontsize
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        Plot.__init__(self, cmap=cmap, orientation=orientation, fontsize=fontsize)
        self._database = Timesteps_Database(dbpath=dbpath, completeTable=completeTable)
 
        self._colorsTimestep = {1: 'C0', 2: 'C1', 4: 'C2', 8: 'C3', 16: 'C4', 32: 'C5', 64: 'C9'}


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self._database.close_connection()


    def plot_spinup_data(self, model, parameterId, ncol=3, subPlot=False, timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, timestepListSubPlot=Metos3d_Constants.METOS3D_TIMESTEPS[:-2], axesResultSmall=[.61, .30, .3, .34], subPlotModelYear=8000, additionalSimulationIds=[]):
        """
        Plot the spinup for the given parameterId and model for all timesteps

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        parameterId : int
            Id of the parameter of the latin hypercube example
        ncol : int, default: 3
            Number of columns for the legend
        subPlot : bool, default: False
            If the value is True, an enlargment of the last 2000 model years
            of the spin up norm is inserted as a extra subplot. For the value
            False, no subplot is added.
        timestepList : list [int], default: metos3dutil.metos3d.constants.
            METOS3D_TIMESTEPS
            Representation of the spin up norm for each specified timestep
        timestepListSubPlot : list [int], default: metos3dutil.metos3d.
            constants.METOS3D_TIMESTEPS[:-2]
            Representation of the spin up norm in the subplot for each
            specified timestep
        axesResultSmall : list [float], default: [.61, .30, .3, .34]
            Dimensions of the subplot
        subPlotModelYear : int, default: 8000
            Start model year for the subplot
        additionalSimulationIds : list [tuple [int]], default: []
            List for additional spin up plots using the simulationId and
            timestep defined in the tuples
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert type(ncol) is int and 0 < ncol
        assert type(subPlot) is bool
        assert type(timestepList) is list
        assert type(timestepListSubPlot) is list
        assert type(axesResultSmall) is list and len(axesResultSmall) == 4
        assert type(subPlotModelYear) is int and 0 <= subPlotModelYear
        assert type(additionalSimulationIds) is list

        if subPlot:
            self.__axesResultSmall = plt.axes(axesResultSmall)

        simulationIdsPlot = []

        #Plot the spin up for the model and parameterId using default initial concentration
        simulationIds = self._database.get_simids_timestep_for_parameter_model(parameterId, model)
        for simulationId, timestep in simulationIds:
            if timestep in timestepList and self._database.get_convergence(simulationId):
                data = self._database.read_spinup_values_for_simid(simulationId)
                try:
                    self._axesResult.plot(data[:,0], data[:,1], color = self._colorsTimestep[timestep], label = r'{}\si{{\Timestep}}'.format(timestep))
                    if subPlot and timestep in timestepListSubPlot:
                        self.__axesResultSmall.plot(data[subPlotModelYear:,0], data[subPlotModelYear:,1], color = self._colorsTimestep[timestep], label = '{}\si{{\Timestep}}'.format(timestep))

                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")

        #Plot the spin up for the extra simulationIds
        for simulationId, timestep in additionalSimulationIds:
            if timestep in timestepList and self._database.get_convergence(simulationId):
                data = self._database.read_spinup_values_for_simid(simulationId)
                try:
                    self._axesResult.plot(data[:,0], data[:,1], color = self._colorsTimestep[timestep], linestyle='dashed')

                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best', ncol=ncol)

        if subPlot:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


    def plot_spinup_data_simIds(self, ncol=3, additionalSimulationIds=[]):
        """
        Plot the spinup for the given simulationIds 

        Parameters
        ----------
        ncol : int, default: 3
            Number of columns for the legend
        additionalSimulationIds : list [tuple], default: []
            List for additional spin up plots using the simulationId and
            label defined in the tuples
        """
        assert type(ncol) is int and 0 < ncol
        assert type(additionalSimulationIds) is list

        #Plot the spin up for the extra simulationIds
        for simulationId, label in additionalSimulationIds:
            if self._database.get_convergence(simulationId):
                data = self._database.read_spinup_values_for_simid(simulationId)
                try:
                    self._axesResult.plot(data[:,0], data[:,1], label=label)

                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best', ncol=ncol)

    def plot_hist_spinup_year(self, timestep, tolerance, model='N', bins=10, lhs=True):
        """
        Plot a histogram of the required model years of the spinup to reach a given tolerance and timestep over all parameter sets of the latin hypercube sample.
        @author: Markus Pfeil
        """
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(tolerance) is float and 0 < tolerance
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(bins) is int and 0 < bins
        assert type(lhs) is bool

        data = self._database.read_spinup_years_for_timestep_and_tolerance(timestep, tolerance, model, lhs=lhs)
        try:
            self.__axesResult.hist(data[:,1], bins=bins)
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Figure with was not created.")


    def plot_hist_spinup_avg_year(self, tolerance, lhs=True):
        """
        Plot a histogram for the average of the required model years over all parameter sets to reach a given tolerance for the spinup.
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        assert type(tolerance) is float and 0 < tolerance
        assert type(lhs) is bool

        def value_for_timestep(array, timestep):
            assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
            j = 0
            while j < len(array) and array[j][0] != timestep:
                j = j + 1
            return 0 if (j >= len(array)) else array[j][1]

        #Get data for all models from the database
        data_N = self._database.read_spinup_avg_year_for_tolerance(tolierance, 'N', lhs=lhs)
        data_NDOP = self._database.read_spinup_avg_year_for_tolerance(tolerance, 'N-DOP', lhs=lhs)
        data_NPDOP = self._database.read_spinup_avg_year_for_tolerance(tolerance, 'NP-DOP', lhs=lhs)
        data_NPZDOP = self._database.read_spinup_avg_year_for_tolerance(tolerance, 'NPZ-DOP', lhs=lhs)
        data_NPZDDOP = self._database.read_spinup_avg_year_for_tolerance(tolerance, 'NPZD-DOP', lhs=lhs)
        data_MITgcmPO4DOP = self._database.read_spinup_avg_year_for_tolerance(tolerance, 'MITgcm-PO4-DOP', lhs=lhs)

        model_count = len(Metos3d_Constants.METOS3D_MODELS)
        timestep_count = len(Metos3d_Constants.METOS3D_TIMESTEPS)
        ind = np.arange(model_count)
        width = 1 / (timestep_count + 1)
        offset = (1 - timestep_count * width) / 2
        i = 0
        leg = []
        rec = []

        for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
            data = np.zeros(6)
            data[0] = value_for_timestep(data_N, timestep) if not (i < len(data_N) and data_N[i][0] == timestep) else data_N[i][1]
            data[1] = value_for_timestep(data_NDOP, timestep) if not (i < len(data_NDOP) and data_NDOP[i][0] == timestep) else data_NDOP[i][1]
            data[2] = value_for_timestep(data_NPDOP, timestep) if not (i < len(data_NPDOP) and data_NPDOP[i][0] == timestep) else data_NPDOP[i][1]
            data[3] = value_for_timestep(data_NPZDOP, timestep) if not (i < len(data_NPZDOP) and data_NPZDOP[i][0] == timestep) else data_NPZDOP[i][1]
            data[4] = value_for_timestep(data_NPZDDOP, timestep) if not (i < len(data_NPZDDOP) and data_NPZDDOP[i][0] == timestep) else data_NPZDDOP[i][1]
            data[5] = value_for_timestep(data_MITgcmPO4DOP, timestep) if not (i < len(data_MITgcmPO4DOP) and data_MITgcmPO4DOP[i][0] == timestep) else data_MITgcmPO4DOP[i][1]

            try:
                rec.append(self.__axesResult.bar(ind + offset + i * width, data, width, color=self._colorsTimestep[timestep]))
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Figure was not created.")
            leg.append('{}\si{\Timestep}'.format(timestep))
            i = i + 1

        #Set labels
        self.__axesResult.set_xticks(ind + timestep_count * width * 0.5)
        self.__axesResult.set_xticklabels(('N', 'N-DOP', 'NP-DOP', 'NPZ-\nDOP', 'NPZD-\nDOP', 'MITgcm-\nPO4-DOP'))
        self.__axesResult.set_ylabel(r'Model years [\si{\Modelyear}]')

        self.__axesResult.legend((rec[0][0], rec[1][0], rec[2][0], rec[3][0], rec[4][0], rec[5][0], rec[6][0]), leg, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., labelspacing=0.1, borderpad=0.2)


    def plot_violinplot_spinup_avg_year(self, model, tolerance=10**(-4), points=100, showmeans=True, showmedians=False, showextrema=False, showquartile=False, lhs=True):
        """
        Plot a violin plot for the average of the years over all parameter sets to reach a given tolerance for the spinup. The plot is for the given model
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(tolerance) is float and 0 < tolerance
        assert type(points) is int and 0 < points
        assert type(showmeans) is bool
        assert type(showmedians) is bool
        assert type(showextrema) is bool
        assert type(showquartile) is bool
        assert type(lhs) is bool

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value
                                       
        #Get data for all parameter indicies from the database
        data = []
        timestepList = []
        for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
            spinupYears  = self._database.read_spinup_years_for_tolerance_timestep(tolerance, model, timestep, lhs=lhs)
            if len(spinupYears) > 0:
                data.append(spinupYears)
                timestepList.append(timestep)

        quartile1 = np.empty(len(timestepList))
        medians = np.empty(len(timestepList))
        quartile3 = np.empty(len(timestepList))
        perzentile05 = np.empty(len(timestepList))
        perzentile95 = np.empty(len(timestepList))
        whiskers = np.empty(len(timestepList))

        for i in range(len(timestepList)):
            perzentile05[i], quartile1[i], medians[i], quartile3[i], perzentile95[i] = np.percentile(data[i], [5, 25, 50, 75, 95], axis=0)

        self.__axesResult.violinplot(data, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema, points=points)

        if (showquartile):
            whiskers = np.array([adjacent_values(data[i], quartile1[i], quartile3[i]) for i in range(len(timestepList))])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self.__axesResult.scatter(inds, medians, marker="_", color='white', s=10, zorder=3)
            self.__axesResult.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=4)
            self.__axesResult.vlines(inds, perzentile05, perzentile95, color='k', linestyle='-', lw=.5)

        #Set labels
        self.__axesResult.set_xlabel(r'Time step [\si{\Timestep}]')
        self.__axesResult.set_ylabel(r'Model years [\si{\Modelyear}]')
        self.__axesResult.set_xticklabels(('', '1', '2', '4', '8', '16', '32', '64'))


    def plot_violinplot_spinup_avg_tolerance(self, model, year, points=100, showmeans=True, showmedians=False, showextrema=False, showquartile=False, lhs=True):
        """
        Plot a violin plot for the average of the tolerance over all parameter sets reaching the spin-up in the given years. The plot is for the given model
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert type(points) is int and 0 < points
        assert type(showmeans) is bool
        assert type(showmedians) is bool
        assert type(showextrema) is bool
        assert type(showquartile) is bool
        assert type(lhs) is bool

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        timestepArray = Metos3d_Constants.METOS3D_TIMESTEPS
        quartile1 = np.empty(len(timestepArray))
        medians = np.empty(len(timestepArray))
        quartile3 = np.empty(len(timestepArray))
        perzentile05 = np.empty(len(timestepArray))
        perzentile95 = np.empty(len(timestepArray))
        whiskers = np.empty(len(timestepArray))

        #Get data for all models from the database
        data = []
        for i in range(len(timestepArray)):
            data.append(sorted(self._database.read_spinup_tolerance_for_year_timestep(model, timestepArray[i], year=year, lhs=lhs)))
            perzentile05[i], quartile1[i], medians[i], quartile3[i], perzentile95[i] = np.percentile(data[i], [5, 25, 50, 75, 95], axis=0)

        self.__axesResult.violinplot(data, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema, points=points)

        if (showquartile):
            whiskers = np.array([adjacent_values(data[i], quartile1[i], quartile3[i]) for i in range(len(timestepArray))])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self.__axesResult.scatter(inds, medians, marker="_", color='white', s=10, zorder=3)
            self.__axesResult.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=4)
            self.__axesResult.vlines(inds, perzentile05, perzentile95, color='k', linestyle='-', lw=.5)

        #Set labels
        self.__axesResult.set_yscale('log', basey=10)
        self.__axesResult.set_xlabel(r'Time step [\si{\Timestep}]')
        self.__axesResult.set_ylabel(r'Spin-up tolerance')
        self.__axesResult.set_xticklabels(('', '1', '2', '4', '8', '16', '32', '64'))


    def plot_tracer_norm_data(self, parameterId, model, norm='2', trajectory='', year=None, ncol=3, additionalSimulationIds=[]):
        """
        Plot the norm of the tracer concentration difference
        
        Plot the development over the spin up (10000 years) of the difference
        between the 1dt solution and coarse solutions (2, 4, 8, 16, 32 and
        64dt) in the norm for the given parameterId and the model for all
        time steps.

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        model : str
            Name of the biogeochemical model
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        year : int, default: None
            Use the reference solution (1dt solution) at the given year (e.g. the reference soluation after a spin up over 10000 model years). If the value is None, use the same year for the reference and coarse solution.
        ncol : int, default: 3
            Number of columns for the legend
        additionalSimulationIds : list [tuple [int]], default: []
            List for additional spin up plots using the simulationId and
            timestep defined in the tuples
        """
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'trajectory']
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol
        assert type(additionalSimulationIds) is list

        simids = self._database.get_simids_timestep_for_parameter_model(parameterId, model)
        #Get 1dt solution vector or value for year 10000
        i = 0
        while simids[i][1] != 1:
            i = i+1
        if year is None:
            data_solution = self._database.read_tracer_norm_values_for_simid(simids[i][0], norm=norm, trajectory=trajectory)
        else:
            data_solution = self._database.read_tracer_norm_value_for_simid_year(simids[i][0], year, norm=norm, trajectory=trajectory)

        #Plot the development of the spin up for 2, 4, 8, 16, 32 and 64dt
        for simulationId, timestep in simids:
            if timestep != 1 and self._database.get_convergence(simulationId):
                data = self._database.read_tracer_difference_norm_values_for_simid(simulationId, simids[i][0], yearB=year, norm=norm, trajectory=trajectory)
                if year is None:
                    data[:,1] = data[:,1] / data_solution[:,1]
                else:
                    data[:,1] = data[:,1] / data_solution
                try:
                    self._axesResult.plot(data[:,0], data[:,1], color=self._colorsTimestep[timestep], label = r'{}\si{{\Timestep}}'.format(timestep))
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Result do not plot")

        #Plot the norm for the extra simulationIds
        for simulationId, timestep in additionalSimulationIds:
            if self._database.get_convergence(simulationId):
                data = self._database.read_tracer_difference_norm_values_for_simid(simulationId, simids[i][0], yearB=year, norm=norm, trajectory=trajectory)
                if year is None:
                    data[:,1] = data[:,1] / data_solution[:,1]
                else:
                    data[:,1] = data[:,1] / data_solution
                try:
                    self._axesResult.plot(data[:,0], data[:,1], color = self._colorsTimestep[timestep], linestyle='dashed')

                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Relative error')
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best', ncol=ncol)


    def plot_tracer_norm_data_simIds(self, parameterId, model, norm='2', trajectory='', year=None, ncol=3, additionalSimulationIds=[]):
        """
        Plot the norm of the tracer concentration difference
        
        Plot the development over the spin up (10000 years) of the difference
        between the 1dt solution and coarse solutions (2, 4, 8, 16, 32 and
        64dt) in the norm for the given parameterId and the model for all
        time steps.

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        model : str
            Name of the biogeochemical model
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'trajectory' the norm over the whole trajectory
        year : int, default: None
            Use the reference solution (1dt solution) at the given year (e.g. the reference soluation after a spin up over 10000 model years). If the value is None, use the same year for the reference and coarse solution.
        ncol : int, default: 3
            Number of columns for the legend
        additionalSimulationIds : list [tuple], default: []
            List for additional spin up plots using the simulationId and
            label defined in the tuples
        """
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'trajectory']
        assert year is None or type(year) is int and 0 <= year
        assert type(ncol) is int and 0 < ncol
        assert type(additionalSimulationIds) is list

        simids = self._database.get_simids_timestep_for_parameter_model(parameterId, model)
        #Get 1dt solution vector or value for year 10000
        i = 0
        while simids[i][1] != 1:
            i = i+1
        if year is None:
            data_solution = self._database.read_tracer_norm_values_for_simid(simids[i][0], norm=norm, trajectory=trajectory)
        else:
            data_solution = self._database.read_tracer_norm_value_for_simid_year(simids[i][0], year, norm=norm, trajectory=trajectory)

        #Plot the norm for the extra simulationIds
        for simulationId, label in additionalSimulationIds:
            if self._database.get_convergence(simulationId):
                data = self._database.read_tracer_difference_norm_values_for_simid(simulationId, simids[i][0], yearB=year, norm=norm, trajectory=trajectory)
                if year is None:
                    data[:,1] = data[:,1] / data_solution[:,1]
                else:
                    data[:,1] = data[:,1] / data_solution
                try:
                    self._axesResult.plot(data[:,0], data[:,1], label=label)

                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure with was not created.")

        #Set labels
        self._axesResult.set_xlabel(r'Model years [\si{{\Modelyear}}]')
        self._axesResult.set_ylabel(r'Relative error')
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best', ncol=ncol)


    def plot_hist_tracer_norm(self, timestep, model='N', year=10000, norm='Boxweighted', endpoint=False, bins=10, lhs=True):
        """
        Plot a histogram of the relative errors (error between coarse und 1dt solution divide through norm values of 1dt solution) for year, timestep and model over all parameter sets
        @author: Markus Pfeil
        """
        assert timestep in [1, 2, 4, 8, 16, 32, 64]
        assert model in ['N', 'N-DOP', 'NP-DOP', 'NPZ-DOP', 'NPZD-DOP', 'MITgcm-PO4-DOP']
        assert year is not None and year >= 0
        assert norm in ['2', 'Boxweighted', 'BoxweightedVol']
        assert endpoint or not endpoint

        data = self.database.read_tracer_norm_rel_error_values_model_timestep(model, timestep, year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)
        try:
            self.__axesResult.hist(data[:,1], bins=bins)
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_hist_norm_avg_year(self, year=10000, norm='Boxweighted', endpoint=False, lhs=True):
        """
        Plot a histogram for the average of the relative error over all parameter sets of the latin hypercube sample for the tracer norm.
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        assert year is not None and year >= 0
        assert norm in ['2', 'Boxweighted', 'BoxweightedVol']
        assert endpoint or not endpoint

        #Get data for all models from the database
        data_N = self.database.read_norm_avg_error_for_model('N', year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)
        data_NDOP = self.database.read_norm_avg_error_for_model('N-DOP', year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)
        data_NPDOP = self.database.read_norm_avg_error_for_model('NP-DOP', year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)
        data_NPZDOP = self.database.read_norm_avg_error_for_model('NPZ-DOP', year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)
        data_NPZDDOP = self.database.read_norm_avg_error_for_model('NPZD-DOP', year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)
        data_MITgcmPO4DOP = self.database.read_norm_avg_error_for_model('MITgcm-PO4-DOP', year=year, norm=norm, difference='Difference', endpoint=endpoint, lhs=lhs)

        model_count = 6
        timestep_count = 6
        ind = np.arange(model_count)
        width = 1 / (timestep_count + 1)
        offset = (1 - timestep_count * width) / 2
        i = 0
        leg = []
        rec = []

        #Skip timestep 1
        i = i+1
        for timestep in [2, 4, 8, 16, 32, 64]:
            assert data_N[i][0] == timestep and data_NDOP[i][0] == timestep and data_NPDOP[i][0] == timestep and data_NPZDOP[i][0] == timestep and data_NPZDDOP[i][0] == timestep and data_MITgcmPO4DOP[i][0] == timestep
            data = np.array([data_N[i][1], data_NDOP[i][1], data_NPDOP[i][1], data_NPZDOP[i][1], data_NPZDDOP[i][1], data_MITgcmPO4DOP[i][1]])
            try:
                rec.append(self.__axesResult.bar(ind + offset + (i-1) * width, data, width, color=self._colors[timestep]))
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Result do not plot")
            leg.append('{}{}'.format(timestep, '\si{\Timestep}'))
            i = i+1

        #Set labels
        self.__axesResult.set_xticks(ind + offset + timestep_count * width * 0.5)
        self.__axesResult.set_xticklabels(('N', 'N-DOP', 'NP-DOP', 'NPZ-\nDOP', 'NPZD-\nDOP', 'MITgcm-\nPO4-DOP'))
        self.__axesResult.set_yscale('log', basey=10)
        self.__axesResult.set_ylabel('Relative error')

        self.__axesResult.legend((rec[0][0], rec[1][0], rec[2][0], rec[3][0], rec[4][0], rec[5][0]), leg, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=4, mode="expand", borderaxespad=0., labelspacing=0.1, borderpad=0.2)


    def plot_hist_norm_trajectory_avg_year(self, year=10000, norm='boxweighted', timestep_1dt=False, lhs=True):
        """
        Plot a histogram for the average of the relative error over all parameter sets of the latin hypercube sample for the tracer norm of the trajectory.
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        #Get data for all models from the database
        data_N = self.database.read_norm_trajectory_avg_error_for_model('N', year=year, norm=norm, difference='Difference', timestep_1dt=timestep_1dt, lhs=lhs)
        data_NDOP = self.database.read_norm_trajectory_avg_error_for_model('N-DOP', year=year, norm=norm, difference='Difference', timestep_1dt=timestep_1dt, lhs=lhs)
        data_NPDOP = self.database.read_norm_trajectory_avg_error_for_model('NP-DOP', year=year, norm=norm, difference='Difference', timestep_1dt=timestep_1dt, lhs=lhs)
        data_NPZDOP = self.database.read_norm_trajectory_avg_error_for_model('NPZ-DOP', year=year, norm=norm, difference='Difference', timestep_1dt=timestep_1dt, lhs=lhs)
        data_NPZDDOP = self.database.read_norm_trajectory_avg_error_for_model('NPZD-DOP', year=year, norm=norm, difference='Difference', timestep_1dt=timestep_1dt, lhs=lhs)
        data_MITgcmPO4DOP = self.database.read_norm_trajectory_avg_error_for_model('MITgcm-PO4-DOP', year=year, norm=norm, difference='Difference', timestep_1dt=timestep_1dt, lhs=lhs)

        model_count = 6
        timestep_count = 6
        ind = np.arange(model_count)
        width = 1 / (timestep_count + 1)
        offset = (1 - timestep_count * width) / 2

        i = 0
        leg = []
        rec = []

        #Skip timestep 1
        i = i+1
        for timestep in [2, 4, 8, 16, 32, 64]:
            assert data_N[i][0] == timestep and data_NDOP[i][0] == timestep and data_NPDOP[i][0] == timestep and data_NPZDOP[i][0] == timestep and data_NPZDDOP[i][0] == timestep and data_MITgcmPO4DOP[i][0] == timestep
            data = np.array([data_N[i][1], data_NDOP[i][1], data_NPDOP[i][1], data_NPZDOP[i][1], data_NPZDDOP[i][1], data_MITgcmPO4DOP[i][1]])
            try:
                rec.append(self.__axesResult.bar(ind + offset + (i-1) * width, data, width, color=self._colors[timestep]))
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Result do not plot")
            leg.append('{}\si{\Timestep}'.format(timestep))
            i = i+1

        #Set labels
        self.__axesResult.set_xticks(ind + offset + timestep_count * width * 0.5)
        self.__axesResult.set_xticklabels(('N', 'N-DOP', 'NP-DOP', 'NPZ-\nDOP', 'NPZD-\nDOP', 'MITgcm-\nPO4-DOP'))
        self.__axesResult.set_yscale('log', basey=10)

        self.__axesResult.legend((rec[0][0], rec[1][0], rec[2][0], rec[3][0], rec[4][0], rec[5][0]), leg, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., labelspacing=0.1, borderpad=0.2)


    def plot_violinplot_norm_avg(self, model, year=10000, norm='Boxweighted', difference='Difference', endpoint=False, points=100, showmeans=True, showmedians=False, showextrema=False, showquartile=False, lhs=True):
        """
        Plot a violin plot for the average relative norm over all parameter sets. The plot is for the given model
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        timestepArray = [2, 4, 8, 16, 32, 64]
        quartile1 = np.empty(len(timestepArray))
        medians = np.empty(len(timestepArray))
        quartile3 = np.empty(len(timestepArray))
        perzentile05 = np.empty(len(timestepArray))
        perzentile95 = np.empty(len(timestepArray))
        whiskers = np.empty(len(timestepArray))

        #Get data for all timesteps from the database
        data = []
        for i in range(len(timestepArray)):
            data.append(self.database.read_rel_norm_for_model_timestep(model, timestepArray[i], year=year, norm=norm, difference=difference, endpoint=endpoint, lhs=lhs))
            perzentile05[i], quartile1[i], medians[i], quartile3[i], perzentile95[i] = np.percentile(data[i], [5, 25, 50, 75, 95], axis=0)

        self.__axesResult.violinplot(data, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema, points=points)

        if (showquartile):
            whiskers = np.array([adjacent_values(data[i], quartile1[i], quartile3[i]) for i in range(len(timestepArray))])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self.__axesResult.scatter(inds, medians, marker="_", color='white', s=5, zorder=3)
            self.__axesResult.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=4)
            self.__axesResult.vlines(inds, perzentile05, perzentile95, color='k', linestyle='-', lw=.5)

        #Set labels
        self.__axesResult.set_yscale('log', basey=10)
        self.__axesResult.set_xlabel(r'Time step [\si{\Timestep}]')
        self.__axesResult.set_ylabel(r'Relative error')
        self.__axesResult.set_xticklabels(('', '2', '4', '8', '16', '32', '64'))


    def plot_costfunction_data(self, parameterId, model, costfunction='OLS', measurement=0, endpoint=False):
        """
        Plot the development over the spinup (10000 years) of the difference between 1dt solution and coarse solutions (2, 4, 8, 16, 32 and 64dt) in the costfunction for the parameter and the model for all timesteps
        Parameters
        ----------
        parameterId : int
            id of the parameter of the latin hypercube example
        model : string
            name of the model
        endpoint : bool
            Parameter difference choose plot a with the difference in every step (endpoint=False) or a plot with the difference to the endpoint of the 1dt solution (endpoint=True)

        @author: Markus Pfeil
        """
        assert parameterId is not None and parameterId >= 0
        assert model in ['N', 'N-DOP', 'NP-DOP', 'NPZ-DOP', 'NPZD-DOP', 'MITgcm-PO4-DOP']
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert measurement is not None and measurement >= 0
        assert endpoint or not endpoint

        simids = self.database.get_simids_timestep_for_parameter_model(parameterId, model)
        #Get 1dt solution vector or value for year 10000
        i = 0
        while simids[i][1] != 1:
            i = i+1
        if endpoint:
            data_solution = self.database.read_costfunction_values_for_simid_year(simids[i][0], 10000, costfunction=costfunction, measurementId=measurement)
        else:
            data_solution = self.database.read_costfunction_values_for_simid(simids[i][0], costfunction=costfunction, measurementId=measurement)

        #Plot the development of the spinup for 2, 4, 8, 16, 32 and 64dt
        for i in range(len(simids)):
            if simids[i][1] != 1 and self.database.get_convergence(simids[i][0]):
                simid = simids[i][0]
                data = self.database.read_costfunction_diff_values_for_simid(simid, costfunction=costfunction, measurementId=measurement, endpoint=endpoint)
                if endpoint:
                    data[:,1] = data[:,1] / data_solution
                else:
                    data[:,1] = data[:,1] / data_solution[:,1]
                try:
                    self.__axesResult.plot(data[:,0], data[:,1], color=self._colors[simids[i][1]], label = r'{:d}{dt}'.format(simids[i][1], dt='\si{\Timestep}'))
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Result do not plot")
        self.__axesResult.set_yscale('log', basey=10)
        self.__axesResult.legend(loc='right', ncol=2)


    def plot_hist_costfunction(self, timestep, model='N', year=10000, costfunction='OLS', measurement=0, endpoint=False, bins=10, lhs=True):
        """
        Plot a histogram of the relative errors (error between coarse und 1dt solution divide through costfunction values of 1dt solution) for year, timestep and model over all parameter sets of the latin hypercube sample
        @author: Markus Pfeil
        """
        assert timestep in [1, 2, 4, 8, 16, 32, 64]
        assert model in ['N', 'N-DOP', 'NP-DOP', 'NPZ-DOP', 'NPZD-DOP', 'MITgcm-PO4-DOP']
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert year is not None and year >= 0
        assert measurement is not None and measurement >= 0
        assert endpoint or not endpoint

        data = self.database.read_costfunction_rel_error_values_model_timestep(model, timestep, year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)
        try:
            self.__axesResult.hist(data[:,1], bins=bins)
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_hist_costfunction_avg_year(self, year=10000, costfunction='OLS', measurement=0, endpoint=False, lhs=True):
        """
        Plot a histogram for the average of the relative error over all parameter sets for the costfunction values.
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        #Get data for all models from the database
        data_N = self.database.read_costfunction_avg_error_for_model('N', year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)
        data_NDOP = self.database.read_costfunction_avg_error_for_model('N-DOP', year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)
        data_NPDOP = self.database.read_costfunction_avg_error_for_model('NP-DOP', year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)
        data_NPZDOP = self.database.read_costfunction_avg_error_for_model('NPZ-DOP', year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)
        data_NPZDDOP = self.database.read_costfunction_avg_error_for_model('NPZD-DOP', year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)
        data_MITgcmPO4DOP = self.database.read_costfunction_avg_error_for_model('MITgcm-PO4-DOP', year=year, costfunction=costfunction, measurementId=measurement, endpoint=endpoint, lhs=lhs)

        model_count = 6
        timestep_count = 6
        ind = np.arange(model_count)
        width = 1 / (timestep_count + 1)
        offset = (1 - timestep_count * width) / 2
        i = 0
        leg = []
        rec = []

        #Skip timestep 1
        i = i+1
        for timestep in [2, 4, 8, 16, 32, 64]:
            assert data_N[i][0] == timestep and data_NDOP[i][0] == timestep and data_NPDOP[i][0] == timestep and data_NPZDOP[i][0] == timestep and data_NPZDDOP[i][0] == timestep and data_MITgcmPO4DOP[i][0] == timestep
            data = np.array([data_N[i][1], data_NDOP[i][1], data_NPDOP[i][1], data_NPZDOP[i][1], data_NPZDDOP[i][1], data_MITgcmPO4DOP[i][1]])
            try:
                rec.append(self.__axesResult.bar(ind + offset + (i-1) * width, data, width, color=self._colors[timestep]))
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Result do not plot")
            leg.append('{}{}'.format('{}'.format(timestep), '\si{\Timestep}'))
            i = i+1

        #Set labels
        self.__axesResult.set_xticks(ind + offset + timestep_count * width * 0.5)
        self.__axesResult.set_xticklabels(('N', 'N-DOP', 'NP-DOP', 'NPZ-\nDOP', 'NPZD-\nDOP', 'MITgcm-\nPO4-DOP'))
        self.__axesResult.set_yscale('log', basey=10)
        self.__axesResult.set_ylabel('Relative cost function value')

        self.__axesResult.legend((rec[0][0], rec[1][0], rec[2][0], rec[3][0], rec[4][0], rec[5][0]), leg, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0., labelspacing=0.1, borderpad=0.2)


    def plot_violinplot_costfunction_avg(self, model, year=10000, costfunction='OLS', measurement=0, endpoint=False, points=100, showmeans=True, showmedians=False, showextrema=False, showquartile=False, lhs=True):
        """
        Plot a violin plot for the average relative cost function over all parameter sets. The plot is for the given model
        The plot is for all timesteps.
        @author: Markus Pfeil
        """
        assert model in ['N', 'N-DOP', 'NP-DOP', 'NPZ-DOP', 'NPZD-DOP', 'MITgcm-PO4-DOP']
        assert year >= 0
        assert costfunction in ['OLS', 'WLS', 'GLS']
        assert measurement is not None and measurement >= 0
        assert endpoint or not endpoint

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value

        timestepArray = [2, 4, 8, 16, 32, 64]
        quartile1 = np.empty(len(timestepArray))
        medians = np.empty(len(timestepArray))
        quartile3 = np.empty(len(timestepArray))
        perzentile05 = np.empty(len(timestepArray))
        perzentile95 = np.empty(len(timestepArray))
        whiskers = np.empty(len(timestepArray))

        #Get data for all timesteps from the database
        data = []
        for i in range(len(timestepArray)):
            data.append(self.database.read_rel_costfunction_for_model_timestep(model, timestepArray[i], year=year, costfunction=costfunction, measurement=measurement, endpoint=endpoint, lhs=lhs))
            perzentile05[i], quartile1[i], medians[i], quartile3[i], perzentile95[i] = np.percentile(data[i], [5, 25, 50, 75, 95], axis=0)

        self.__axesResult.violinplot(data, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema, points=points)

        if (showquartile):
            whiskers = np.array([adjacent_values(data[i], quartile1[i], quartile3[i]) for i in range(len(timestepArray))])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self.__axesResult.scatter(inds, medians, marker="_", color='white', s=5, zorder=3)
            self.__axesResult.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=4)
            self.__axesResult.vlines(inds, perzentile05, perzentile95, color='k', linestyle='-', lw=.5)

        #Set labels
        self.__axesResult.set_yscale('log', basey=10)
        self.__axesResult.set_xlabel(r'Time step [\si{\Timestep}]')
        self.__axesResult.set_xticklabels(('', '2', '4', '8', '16', '32', '64'))


    def plot_negative_concentration_count(self, parameterId, model, yearEnd, yearStart=0, ncol=3, loc=0, dataIndex=1, subplots=False, plotMass=False):
        """
        Plot the count of the negative concentrations for the spinup
        @author: Markus Pfeil
        """
        for timestep in [64]: #1, 2, 4, 8, 16, 32, 64
            nsplit = int((2880 / timestep) * yearEnd)
            print('NegativeConcentration: {}, Parameter {:0>3d}, {:d}dt'.format(model, parameterId, timestep))
            data = negativeConcentrations.calculateNegativeConcentrations(model, parameterId, timestep, yearEnd, yearStart=yearStart)
            vol = negativeConcentrations.readBoxVolumes(path='/sfs/fs5/home-sh/sunip350/.metos3d/data/data/TMM/2.8/Geometry/volumes.petsc')
            overallMass = np.sum((2.17 + 10**(-4) * (len(self._TRACER[model]) - 1)) * vol)

            for i in range(np.shape(data)[0]):
                print('{:>4d}:  {}  /  {}  /  {}'.format(i, data[i, 0, 4], data[i, 1, 4], data[i, 2, 4]))
            try:
                for trac in range(len(self._TRACER[model])):
                    if subplots:
                        self.__axesResult[0].plot(data[:nsplit+1,trac,0], data[:nsplit+1,trac,dataIndex], label = '{}'.format(self._TRACER[model][trac]), color= self._colors[2**(trac)])
                        self.__axesResult[1].plot(data[nsplit:,trac,0], data[nsplit:,trac,dataIndex], label = '{}'.format(self._TRACER[model][trac]), color= self._colors[2**(trac)])
                        if plotMass:
                            self.__axesResult[0].plot(data[:nsplit+1,trac,0], np.abs(data[:nsplit+1,trac,5]) / overallMass, color= self._colors[2**(trac)], ls='--')
                            self.__axesResult[1].plot(data[nsplit:,trac,0], np.abs(data[nsplit:,trac,5]) / overallMass, color= self._colors[2**(trac)], ls='--')
                    else:
                        self.__axesResult.plot(data[:,trac,0], np.abs(data[:,trac,dataIndex]) / overallMass, label = '{}'.format(self._TRACER[model][trac]), color= self._colors[2**(trac)])
                        if plotMass:
                            self.__axesResult.plot(data[:,trac,0], np.abs(data[:,trac,5]) / overallMass, color= self._colors[2**(trac)], ls='--')
                #self.__axesResult.plot(data[:,0,0], (np.abs(data[:,0,dataIndex])+np.abs(data[:,1,dataIndex])+np.abs(data[:,2,dataIndex])) / overallMass, label = 'Overall') #'{}'.format(self._TRACER[model][0])) #color= self._colors[0],

                if subplots:
                    self.__axesResult[0].set_xlim(right=data[nsplit,trac,0])
                    self.__axesResult[1].set_xlim(left=data[nsplit,trac,0])
                    self.__axesResult[1].label_outer()
                    self.__axesResult[0].legend(loc=loc, ncol=ncol, columnspacing=0.9) #, columnspacing=0.9
                else:
                    self.__axesResult.legend(loc=loc, ncol=ncol)
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Result do not plot")


    def plot_spinup_norm(self, parameterId, model, timestep, yearStart=1, yearEnd=5000, steps=1):
        """
        Plot the spin-up norm for the convergence analysis.
        @author: Markus Pfeil
        """
        pathLhs = '/sfs/fs2/work-sh1/sunip350/metos3d/LatinHypercubeSample/Timesteps'
        path = os.path.join(pathLhs, model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(timestep), 'ConvergenceAnalysis', 'Tracer') #TODO: TracerOneStep

        data = np.zeros(shape=((yearEnd-yearStart)//steps, 2))

        tracer = negativeConcentrations.getTracerOutput(path, model, yearStart, 179)
        #tracer = negativeConcentrations.getTracerOutput(path, model, yearStart, None)

        #pathSolution = os.path.join(pathLhs, model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(1), 'TracerOnestep')
        #tracerSolution = negativeConcentrations.getTracerOutput(pathSolution, model, 10000, None)

        i = 0
        for year in range(yearStart+1, yearEnd, steps):
            #TODO: Tracer aus OneStep-Berechnung einlesen
            tracerNext = negativeConcentrations.getTracerOutput(path, model, year, 179)
            #tracerNext = negativeConcentrations.getTracerOutput(path, model, year, None)

            data[i, 0] = year
            data[i, 1] = np.sqrt(np.sum((tracer - tracerNext)**2))
            i = i+1

            tracer = tracerNext

        try:
            self.__axesResult.plot(data[:,0], data[:,1], color= self._colors[timestep], label = '{}dt'.format(timestep))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")
        self.__axesResult.set_yscale('log', basey=10)
        #self.__axesResult.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, labelspacing=0.25, borderpad=0.25, ncol=3)
        self.__axesResult.legend(loc=0)


    def plot_r_convergence(self, parameterId, model, timestep, yearStart=1, yearEnd=5000, tracer_length=52749):
        """
        Plot the r-convergence for the convergence analysis.
        @author: Markus Pfeil
        """
        pathLhs = '/sfs/fs2/work-sh1/sunip350/metos3d/LatinHypercubeSample/Timesteps'
        path = os.path.join(pathLhs, model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(timestep), 'ConvergenceAnalysis', 'TracerOnestep')

        tracer = 10**(-4) * np.ones(shape=(len(self._TRACER[model]), tracer_length))
        tracer[0,:] = 2.17
        for year in range(yearStart, yearEnd):
            #TODO: Tracer aus OneStep-Berechnung einlesen
            tracerYear = negativeConcentrations.getTracerOutput(path, model, year, None)

            data[year, 0] = year
            data[year, 1] = (np.sum((tracerYear - tracerSolution)**2) / np.sum((tracer - tracerSolution)**2))**(1.0/(1.0 *year))

        try:
            self.__axesResult.plot(data[yearStart+1:,0], data[yearStart+1:,1], color= self._colors[timestep], label = '{}dt'.format(timestep))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")
        self.__axesResult.set_yscale('log', basey=10)
        #self.__axesResult.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, labelspacing=0.25, borderpad=0.25, ncol=3)
        self.__axesResult.legend(loc=0)


    def plot_q_convergence(self, parameterId, model, timestep, yearStart=1, yearEnd=5000, tracer_length=52749):
        """
        Plot the q-convergence for the convergence analysis.
        @author: Markus Pfeil
        """
        pathLhs = '/sfs/fs2/work-sh1/sunip350/metos3d/LatinHypercubeSample/Timesteps'
        path = os.path.join(pathLhs, model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(timestep), 'ConvergenceAnalysis', 'TracerOnestep')

        data = np.zeros(shape=(yearEnd-yearStart, 2))
        pathSolution = os.path.join(pathLhs, model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(1), 'TracerOnestep')
        tracerSolution = negativeConcentrations.getTracerOutput(pathSolution, model, 10000, None)
        tracer = negativeConcentrations.getTracerOutput(path, model, yearStart, None)
        for year in range(yearStart+1, yearEnd):
            tracerNext = negativeConcentrations.getTracerOutput(path, model, year, None)

            data[year, 0] = year
            data[year, 1] = np.sum((tracerNext - tracerSolution)**2) / np.sum((tracer - tracerSolution)**2)

            tracer = tracerNext

        try:
            self.__axesResult.plot(data[yearStart+1:,0], data[yearStart+1:,1], color= self._colors[timestep], label = '{}dt'.format(timestep))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")
        #self.__axesResult.set_yscale('log', basey=10)
        #self.__axesResult.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, labelspacing=0.25, borderpad=0.25, ncol=3)
        self.__axesResult.legend(loc=0)
        data = np.zeros(shape=(yearEnd-yearStart, 2))
        pathSolution = os.path.join(pathLhs, model, 'Parameter_{:0>3d}'.format(parameterId), '{:d}dt'.format(1), 'TracerOnestep')
        tracerSolution = negativeConcentrations.getTracerOutput(pathSolution, model, 10000, None)


    def plot_scatter_spinup_norm(self, model, year=10000, norm='2', trajectory='', timestepList=Metos3d_Constants.METOS3D_TIMESTEPS[1:], alpha=0.75):
        """
        Scatter plot of the spin up against the norm

        Plot a scatter plot of the relation between the spin up norm and the
        norm of the tracer difference between the coarse spin up calculation
        and the spin up calculation using the time step 1dt. The plot 
        visualizes the ratio for the given time steps using different colors.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'Trajectory' the norm over the whole trajectory
        timestepList : list [int], default: metos3dutil.metos3d.constants.
            METOS3D_TIMESTEPS[1:]
            Representation of the relation between spin up norm and the norm of
            tracer difference for each specified timestep
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert trajectory in ['', 'Trajectory']
        assert type(timestepList) is list
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0

        oscillationLegend = True

        for timestep in timestepList:
            data = self._database.read_spinupNom_relNorm_for_model_year(model, timestep, year=year, norm=norm, trajectory=trajectory, lhs=False)

            #Determine oscillation of the spin up norm
            oscillation = []
            for i in range(len(data)):
                oscillation.append(self._database.oscillation_spin(int(data[i,0])))
            oscillation = np.array(oscillation)

            try:
                p1 = self._axesResult.scatter(data[:,2][np.invert(oscillation)], data[:,3][np.invert(oscillation)], s=4, marker='.', color=self._colorsTimestep[timestep], alpha=alpha, label = r'{}\si{{\Timestep}}'.format(timestep))

                #Use different markers for spin up calculations with an oscillating spin up norm
                if oscillation.sum() > 0:
                    p2 = self._axesResult.scatter(data[:,2][oscillation], data[:,3][oscillation], s=4, marker='1', color=self._colorsTimestep[timestep], alpha=alpha)
                    #Create legend for different markers
                    if oscillationLegend:
                        legendOscillation = self._axesResult.legend([p1, p2], ['No', 'Yes'], loc='lower left' if model == 'NP-DOP' else 'lower right', title='Osci.', borderaxespad=0.25, labelspacing=0.25, borderpad=0.25, handlelength=0.4, handletextpad=0.5)

                        #Change color of the markers to black
                        lh = legendOscillation.legendHandles
                        for i in range(len(lh)):
                            lh[i].set_color('black')

                        self._axesResult.add_artist(legendOscillation)
                        oscillationLegend = False

                self._axesResult.set_xscale('log', basex=10)
                self._axesResult.set_yscale('log', basey=10)
                self._axesResult.set_xlabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
                self._axesResult.set_ylabel(r'Relative error')
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Figure was not created.")


    def plot_scatter_error_reduction(self, model, year=10000, norm='2', timestepList=Metos3d_Constants.METOS3D_TIMESTEPS[1:], alpha=0.75):
        """
        Scatter plot of the spin up against the error reduction

        Scatter plot of the relation between the spin up norm and the
        reduction of the relative error (norm of the tracer difference
        between the spin up calculation using the coarse time step and the
        spin up calculation using the time step 1dt) for the initial
        concentration and the given model year. The plot visualizes the ratio
        for the given time steps using different colors.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        norm : string, default: '2'
            Used norm
        timestepList : list [int], default: metos3dutil.metos3d.constants.
            METOS3D_TIMESTEPS[1:]
            Representation of the relation between spin up norm and the norm of
            tracer difference for each specified timestep
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert norm in DB_Constants.NORM
        assert type(timestepList) is list
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0

        oscillationLegend = True

        #Set weights for the norm calculation
        if norm == '2':
            normWeight = np.ones(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        elif norm == 'Boxweighted':
            normvol = readBoxVolumes(normvol=True)
            normWeight = np.empty(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
            for i in len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]):
                normWeight[i,:] = normvol
        elif norm == 'BoxweightedVol':
            vol = readBoxVolumes()
            normWeight = np.empty(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
            for i in len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]):
                normWeight[i,:] = vol

        #Initial tracer concentration
        initialTracer = np.reshape(np.array(Metos3d_Constants.INITIAL_CONCENTRATION[model]), (len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]), 1)) * np.ones(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]), Metos3d_Constants.METOS3D_VECTOR_LEN))

        for timestep in timestepList:
            data = self._database.read_spinupNom_relNorm_for_model_year(model, timestep, year=year, norm=norm, lhs=False)

            #Calculate relative error of the initial concentration
            relErrorInitial = np.zeros(len(data))
            tracerRef = np.empty(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
            for i in range(len(data)):
                #Read tracer concentration of the spin up using time step 1 dt
                j = 0
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
                    filename = os.path.join(Timesteps_Constants.PATH, 'Timesteps', model, 'Parameter_{:0>3d}'.format(int(data[i,1])), '{:d}dt'.format(1), 'TracerOnestep', Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, tracer))
                    tracerRef[j, :] = petsc.readPetscFile(filename)
                    j = j + 1

                #Calculate relative error
                relErrorInitial[i] = np.sqrt(np.sum((initialTracer - tracerRef)**2 * normWeight))

            #Determine oscillation of the spin up norm
            oscillation = []
            for i in range(len(data)):
                oscillation.append(self._database.oscillation_spin(int(data[i,0])))
            oscillation = np.array(oscillation)

            try:
                p1 = self._axesResult.scatter(data[:,2][np.invert(oscillation)], data[:,3][np.invert(oscillation)] / relErrorInitial[np.invert(oscillation)], s=4, marker='.', color=self._colorsTimestep[timestep], alpha=alpha, label = r'{}\si{{\Timestep}}'.format(timestep))

                #Use different markers for spin up calculations with an oscillating spin up norm
                if oscillation.sum() > 0:
                    p2 = self._axesResult.scatter(data[:,2][oscillation], data[:,3][oscillation] / relErrorInitial[oscillation], s=4, marker='1', color=self._colorsTimestep[timestep], alpha=alpha)
                    #Create legend for different markers
                    if oscillationLegend:
                        legendOscillation = self._axesResult.legend([p1, p2], ['No', 'Yes'], loc='lower left' if model == 'NP-DOP' else 'lower right', title='Osci.', borderaxespad=0.25, labelspacing=0.25, borderpad=0.25, handlelength=0.4, handletextpad=0.5)

                        #Change color of the markers to black
                        lh = legendOscillation.legendHandles
                        for i in range(len(lh)):
                            lh[i].set_color('black')

                        self._axesResult.add_artist(legendOscillation)
                        oscillationLegend = False

                self._axesResult.set_xscale('log', basex=10)
                self._axesResult.set_yscale('log', basey=10)
                self._axesResult.set_xlabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
                self._axesResult.set_ylabel(r'Relative error reduction')
            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Figure was not created.")


    def plot_scatter_required_model_years(self, model, tolerance=0.0001, norm='2', trajectory='', timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, alpha=0.75):
        """
        Scatter plot of the norm against the required model years

        Scatter plot of the relation between the norm (norm of tracer
        concentration difference between the solution of the spin up using the
        given tolerance and the reference solution (using 1dt over 10000 model
        years) and the required model years to reach the given tolerance. The
        plot visualizes the ratio for the given time steps using different
        colors.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        tolerance : float, default: 0.0001
            Tolerance of the spin up
        norm : string, default: '2'
            Used norm
        trajectory : str, default: ''
            Use for '' the norm only at the first time point in a model year
            and use for 'Trajectory' the norm over the whole trajectory
        timestepList : list [int], default: metos3dutil.metos3d.constants.
            METOS3D_TIMESTEPS
            Representation of the relation between spin up norm and the norm of
            tracer difference for each specified timestep
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(tolerance) is float and 0.0 <= tolerance
        assert norm in DB_Constants.NORM
        assert type(timestepList) is list
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0

        for timestep in timestepList:
            data = self._database.read_tolerance_required_ModelYears_relNorm(model, timestep, tolerance=tolerance, norm=norm, trajectory=trajectory, lhs=False)

            if len(data) > 0:
                try:
                    self._axesResult.scatter(data[:,3], data[:,2], s=4, marker='.', color=self._colorsTimestep[timestep], alpha=alpha, label = r'{}\si{{\Timestep}}'.format(timestep))

                    self._axesResult.set_xscale('log', basex=10)
                    self._axesResult.set_xlabel(r'Relative error')
                    self._axesResult.set_ylabel(r'Model years [\si{{\Modelyear}}]')
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure was not created.")


    def plot_scatter_costfunction(self, model, year=10000, costfunction='OLS', measurementId=0, timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, alpha=0.75):
        """
        Scatter plot of spin-up tolerance against the cost function value

        Scatter plot of the relation between the spin-up tolerance and the cost
        function value. The plot visualizes the ratio for the given time steps
        using different colors.

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        year : int, default: 10000
            Used model year of the spin up (for the spin up is the previous
            model year used)
        costfunction : {'OLS', 'GLS', 'WLS'}, default: 'OLS'
            Type of the cost function
        measurementId : int, default: 0
            Selection of the tracer included in the cost function calculation
        timestepList : list [int], default: metos3dutil.metos3d.constants.
            METOS3D_TIMESTEPS
            Representation of the relation between spin up norm and the norm of
            tracer difference for each specified timestep
        alpha : float, default: 0.75
            The alpha blending value of the scatter plot, between 0
            (transparent) and 1 (opaque)
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'GLS', 'WLS']
        assert type(measurementId) is int and 0 <= measurementId
        assert type(timestepList) is list
        assert type(alpha) is float and 0.0 <= alpha and alpha <= 1.0

        for timestep in timestepList:
            data = self._database.read_costfunction_relNorm(model, timestep, year=year, costfunction=costfunction, measurementId=measurementId, lhs=False)

            if len(data) > 0:
                try:
                    self._axesResult.scatter(data[:,3], data[:,2], s=4, marker='.', color=self._colorsTimestep[timestep], alpha=alpha, label = r'{}\si{{\Timestep}}'.format(timestep))

                    self._axesResult.set_xscale('log', basex=10)
                    self._axesResult.set_yscale('log', basey=10)
                    self._axesResult.set_xlabel(r'Norm [\si{\milli\mole\Phosphat\per\cubic\meter}]')
                    self._axesResult.set_ylabel(r'Cost functional $J_{\text{OLS}}$')
                except IOError as e:
                    print("Error message: " + e.args[1])
                    print("Error message: Figure was not created.")


    def plot_oscillation_model_parameter(self, model, timestep):
        """
        Scatter plot of the model parameter for the oscillating spin up norms

        Parameters
        ----------
        model : str
            Name of the biogeochemical model
        timestep : int
            Timestep used for the spin up calculation 
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        #Read parameter values of the oscillating spin up norms
        modelList = []
        simulationIdList = self._database.get_simids_for_model_timestep(model, timestep)

        for simulationId in simulationIdList:
            if self._database.oscillation_spin(simulationId):
                parameterId = self._database.get_parameterId_for_simid(simulationId)
                modelList.append(self._database.get_parameter(parameterId, model))

        ub = Metos3d_Constants.UPPER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[model]]
        lb = Metos3d_Constants.LOWER_BOUND[Metos3d_Constants.PARAMETER_RESTRICTION[model]]

        #Model parameter b_D of the NPZD-DOP model ist always 0.0
        if model == 'NPZD-DOP':
            ub[-1] = 1.0

        data = np.zeros(shape=(len(modelList) * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model],2))
        color = np.zeros(len(modelList) * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model])

        for i in range(len(modelList)):
            data[i * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model]:(i+1) * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model], 0] = np.arange(Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model])
            data[i * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model]:(i+1) * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model], 1] = (modelList[i] - lb) / (ub - lb)
            color[i * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model]:(i+1) * Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model]] = i/len(modelList)

        self._axesResult.scatter(data[:,0], data[:,1], c=color, s=4, marker='.')
        self._axesResult.set_xticks(np.arange(Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model]))
        self._axesResult.set_xticklabels([r'${}$'.format(i) for i in Metos3d_Constants.PARAMETER_NAMES_LATEX[Metos3d_Constants.PARAMETER_RESTRICTION[model]].tolist()])


