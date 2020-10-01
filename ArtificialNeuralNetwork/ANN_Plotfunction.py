#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

from metos3dutil.plot.plot import Plot
import ann.network.constants as ANN_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.database.constants as DB_Constants
import ann.database.constants as ANN_DB_Constants
from ann.database.access import Ann_Database


class ANN_Plot(Plot):
    def __init__(self, orientation='lc1', fontsize=8, dbpath=ANN_DB_Constants.dbPath, completeTable=True):
        """
        Initialise the database with the simulation data
        @author: Markus Pfeil
        """ 
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        Plot.__init__(self, orientation=orientation, fontsize=fontsize)
        
        self.database = Ann_Database(dbpath=dbpath, completeTable=completeTable)
        #Using color palette used by Vega (https://github.com/vega/vega/wiki/Scales#scale-range-literals): category10 and part of cartegory20
        self._colorsTimestep = {1: 'C0', 2: 'C1', 4: 'C2', 8: 'C3', 16: 'C4', 32: 'C5', 64: 'C9'}


    def closeDatabaseConnection(self):
        """
        Close the connection of the database

        @author: Markus Pfeil
        """
        self.database.close_connection()


    def get_annModel(self, annId):
        """
        Return the model for the given annId
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        
        annType, annModel = self.database.get_annTypeModel(annId)
        return annModel


    def plot_training_loss(self, annId, colors=['r', 'b']):
        """
        Plot the training and validation loss during the training of the ANN.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(colors) is list and len(colors) == 2

        data = self.database.read_losses(annId)
        try:
            self._axesResult.plot(data[:,0], data[:,1], color=colors[0], label='Training loss')
            self._axesResult.plot(data[:,0], data[:,2], color=colors[1], label='Validation loss')

            #Set label
            self._axesResult.set_xlabel(r'Epoch')
            self._axesResult.set_ylabel(r'Loss')
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")
        self._axesResult.set_yscale('log', basey=10)
        self._axesResult.legend(loc='best')


    def plot_spinup_data(self, annId, parameterId, massAdjustment=False, tolerance=None, cpus=64, colors=['r', 'b']):
        """
        Plot the spinup for the spin up using spin up tolerance using constant as well as the ANN prediction as initial value for the given parameter
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert type(colors) is list and len(colors) == 2
        
        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        concentrationIdRef = self.database.get_concentrationId_constantValues(model, Metos3d_Constants.INITIAL_CONCENTRATION[model])
        
        simids = []
        labels = []
        simids.append(self.database.get_simulationId(model, parameterId, concentrationIdRef, False, tolerance=tolerance))
        labels.append('Constant inital value')
        simids.append(self.database.get_simulationId(model, parameterId, concentrationId, massAdjustment, tolerance=tolerance, cpus=cpus)) 
        labels.append('Mass-corrected prediction as initial value')
        
        for i in range(len(simids)):
            data = self.database.read_spinup_values_for_simid(simids[i])
            try:
                self._axesResult.plot(data[:,0], data[:,1], color= colors[i], label = labels[i])

                #Set label
                self._axesResult.set_xlabel(r'Model years [\si{\Modelyear}]')
                self._axesResult.set_ylabel(r'Norm [\si{\milli \mole \Phosphat \per \cubic \metre}]')

            except IOError as e:
                print("Error message: " + e.args[1])
                print("Error message: Result do not plot")
        self._axesResult.set_yscale('log', basey=10)
        #self._axesResult.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, labelspacing=0.25, borderpad=0.25, ncol=3)
        self._axesResult.legend(loc='best')


    def plot_scatter_rel_norm(self, annId, massAdjustment=False, tolerance=None, cpus=64, year=None, norm='2', alpha=0.75, rmax=None, rposition=60, color='b'):
        """
        Plot a scatter polar plot of the relative errors for the given parameters (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert year is None or year >= 0
        assert norm in DB_Constants.NORM
        assert type(alpha) is float
        assert rmax is None or type(rmax) is float and 0 < rmax
        assert type(rposition) is int and 0 <= rposition and rposition <= 360

        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        data = self.database.read_rel_norm(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus, year=year, norm=norm)
        phi = (3.0 * np.pi) / 4.0 + (1 / len(data)) * 2 * np.pi * np.arange(len(data))
        try:
            self._axesResult.scatter(phi, data, s=6, color=color, alpha=alpha)
            ax = self._fig.gca()
            if rmax is not None:
                ax.set_rmax(rmax)

            ax.xaxis.grid(False)
            ax.set_xticklabels([])
            ax.set_rlabel_position(rposition)
            if rmax is not None and rmax < 0.15:
                ax.set_rgrids([0.00025, 0.0005, 0.00075, 0.001], labels=['','0.0005', '', '0.001'])
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_scatter_rel_norm_II(self, data, cmap='hsv', alpha=0.75):
        """
        Plot a scatter polar plot of the relative errors for the given parameters (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample
        @author: Markus Pfeil
        """
        phi = (1 / len(data)) * 2 * np.pi * np.arange(len(data))
        try:
            self._axesResult.scatter(phi, data, s=10, c=data, cmap=cmap, alpha=alpha)
            ax = self._fig.gca()
            #ax.set_rmax(0.55)

            ax.set_xticklabels([])
            ax.set_rlabel_position(75)
            #ax.set_yscale('log')
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_hist_rel_norm(self, annId, massAdjustment=False, tolerance=None, cpus=64, year=None, norm='2', bins=10):
        """
        Plot a histogram of the relative errors for the given parameters (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert year is None or year >= 0
        assert norm in DB_Constants.NORM
        assert bins > 1

        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        data = self.database.read_rel_norm(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus, year=year, norm=norm)
        try:
            self._axesResult.hist(data, bins=bins)
            ax = self._fig.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_hist_mass(self, annId, massAdjustment=False, tolerance=None, cpus=64, year=0, bins=10):
        """
        Plot a histogram of the mass ratio for the given parameter (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert year is None or year >= 0
        assert bins > 1

        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        data = self.database.read_mass(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus, year=year)
        
        data = data * 100 #Conversion into percent
        try:
            self._axesResult.hist(data, bins=bins)
            ax = self._fig.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_hist_spinup_tolerance(self, annId, massAdjustment=False, tolerance=None, cpus=64, year=0, bins=10):
        """
        Plot a histogram of the spin up tolerance for the given parameter (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample.
        Year: None: Use the last year of the spin up. Otherwise use the given year.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert year is None or year >= 0
        assert bins > 1

        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        data = self.database.read_spinup_tolerance(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus, year=year)
        try:
            self._axesResult.hist(data, bins=bins)
            ax = self._fig.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_hist_spinup_year(self, annId, massAdjustment=False, tolerance=None, cpus=64, bins=10, alpha=0.5, maincolor='r', color='b'):
        """
        Plot a histogram of the required years to reach the given spin up tolerance for the given parameter (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert bins > 1

        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        data = self.database.read_spinup_year(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus)
        concentrationIdRef = self.database.get_concentrationId_constantValues(model, Metos3d_Constants.INITIAL_CONCENTRATION[model])
        dataReference = self.database.read_spinup_year(model, concentrationIdRef, tolerance=tolerance, cpus=64)
        try:
            self._axesResult.hist(dataReference[1,:], bins=bins, alpha=alpha, color=maincolor, label='Constant initial value')
            self._axesResult.hist(data[1,:], bins=bins, alpha=alpha, color=color, label='Mass-corrected prediction as initial value')
            
            ax = self._fig.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            #Set label
            self._axesResult.set_xlabel(r'Model years [\si{\Modelyear}]')
            self._axesResult.set_ylabel(r'Occurrence')
            
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")
        self._axesResult.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, labelspacing=0.25, borderpad=0.25)


    def plot_hist_rel_spinup_year(self, annId, massAdjustment=False, tolerance=None, cpus=64, bins=10):
        """
        Plot a histogram of the percentage reduction of the required years to reach the given spin up tolerance for the given parameter (annId, massAdjustment, tolerance and cpus) over all parameter sets of the latin hypercube sample.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert bins > 1

        model = self.get_annModel(annId)
        concentrationId = self.database.get_concentrationId_annValues(annId)
        data = self.database.read_spinup_year(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus)
        concentrationId = self.database.get_concentrationId_constantValues(model, Metos3d_Constants.INITIAL_CONCENTRATION[model])
        dataReference = self.database.read_spinup_year(model, concentrationId, tolerance=tolerance, cpus=64)
        
        assert np.shape(data) == np.shape(dataReference)
        for i in range(np.shape(data)[1]):
            assert data[0,i] == dataReference[0,i]  #ParameterId must be the same
        plotData = np.zeros(shape=(2,np.shape(data)[1]))
        plotData[0,:] = data[0,:]
        plotData[1,:] = (data[1,:] / dataReference[1,:]) * 100
        
        for i in range(np.shape(data)[1]):
            if plotData[1,i] > 100:
                print('ParameterId: {:d}: {}'.format(int(plotData[0,i]), plotData[1,i]))
                
        print('Mean: {}'.format(np.mean(plotData[1,:])))
        try:
            self._axesResult.hist(plotData[1,:], bins=bins)
            plt.axvline(np.mean(plotData[1,:]), color='k', linestyle='dashed', linewidth=1)
            _, max_ = plt.ylim()
            plt.text(np.mean(plotData[1,:]) + 3, max_ - max_/10, 'Mean: {:.2f}'.format(np.mean(plotData[1,:])))
            
            ax = self._fig.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except IOError as e:
            print("Error message: " + e.args[1])
            print("Error message: Result do not plot")


    def plot_violinplot_spinup_avg_year(self, annIdList, massAdjustment=False, tolerance=None, cpus=64, points=100, showmeans=True, showmedians=False, showextrema=False, showquartile=False, colors=None):
        """
        Plot a violin plot for the average of the required years to reach a given tolerance for the spinup over all parameter sets.
        The plot is for the given annIds. 
        @author: Markus Pfeil
        """
        assert len(annIdList) > 0
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(cpus) is int and cpus > 0
        assert colors is None or type(colors) is list
        
        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value
        
        #Get data for all parameter indicies from the database
        data = []
        violinList = []
        model = self.get_annModel(annIdList[0])
        
        #Data set for the reference calculation using constant initial values
        concentrationIdRef = self.database.get_concentrationId_constantValues(model, Metos3d_Constants.INITIAL_CONCENTRATION[model])
        dataReference = self.database.read_spinup_year(model, concentrationIdRef, tolerance=tolerance, cpus=64)[1,:]
        if len(dataReference) > 0:
            data.append(dataReference)
            violinList.append(concentrationIdRef)
        
        for annId in annIdList:
            #Data set for every calculations initialized using the prediction of the ANN with the given annId
            assert annId in range(0, ANN_Constants.ANNID_MAX+1)
            assert model == self.get_annModel(annId)
            concentrationId = self.database.get_concentrationId_annValues(annId)
            dataYear = self.database.read_spinup_year(model, concentrationId, massCorrection=massAdjustment, tolerance=tolerance, cpus=cpus)[1,:]
            if len(dataYear) > 0:
                data.append(dataYear)
                violinList.append(concentrationId)

        quartile1 = np.empty(len(violinList))
        medians = np.empty(len(violinList))
        quartile3 = np.empty(len(violinList))
        perzentile05 = np.empty(len(violinList))
        perzentile95 = np.empty(len(violinList))
        whiskers = np.empty(len(violinList))

        for i in range(len(violinList)):
            perzentile05[i], quartile1[i], medians[i], quartile3[i], perzentile95[i] = np.percentile(data[i], [5, 25, 50, 75, 95], axis=0)

        violin = self._axesResult.violinplot(data, showmeans=showmeans, showmedians=showmedians, showextrema=showextrema, points=points)
        
        #Change the color of the violins
        if colors is not None and len(colors) > len(annIdList):
            for i in range(len(annIdList)+1):
                violin['bodies'][i].set_color(colors[i])
        
        if (showquartile):
            whiskers = np.array([adjacent_values(data[i], quartile1[i], quartile3[i]) for i in range(len(violinList))])
            whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

            inds = np.arange(1, len(medians) + 1)
            self._axesResult.scatter(inds, medians, marker="_", color='white', s=10, zorder=3)
            self._axesResult.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=4)
            self._axesResult.vlines(inds, perzentile05, perzentile95, color='k', linestyle='-', lw=.5)

        #Set labels
        self._axesResult.set_xlabel(r'Initial value')
        self._axesResult.set_ylabel(r'Model years [\si{\Modelyear}]')
        self._axesResult.set_xticks(np.arange(1,len(annIdList)+2))
        self._axesResult.set_xticklabels(('Constant', 'FCN', 'SET', 'GA', 'SET2', 'Conserve'))
