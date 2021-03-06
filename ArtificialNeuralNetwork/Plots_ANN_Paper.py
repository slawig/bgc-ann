#!/usr/bin/env python
# -*- coding: utf8 -*

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import ann.network.constants as ANN_Constants
from metos3dutil.plot.surfaceplot import SurfacePlot
import metos3dutil.petsc.petscfile as petsc
import metos3dutil.metos3d.constants as Metos3d_Constants
from ann.database.access import Ann_Database
from ANN_Plotfunction import ANN_Plot

#Global variables
PATH_DATABASE = '/gxfs_work1/cau/sunip350/metos3d/ArtificialNeuralNetwork/PaperData/ANN_Database.db'
PATH_ANN_RESULTS = '/gxfs_work1/cau/sunip350/metos3d/ArtificialNeuralNetwork/PaperData/ANN-Results'
PATH_REFERENCE_RESULTS = '/gxfs_work1/cau/sunip350/metos3d/ArtificialNeuralNetwork/PaperData/Reference-Results'
PATH_FIGURE = '/gxfs_work1/cau/sunip350/metos3d/ArtificialNeuralNetwork/PaperData/Figures'


#Pattern for the filenames of the tracer concentration
PATTERN_PREDICTION = 'Prediction_{}.petsc'
PATTERN_MASSCORRECTED_PREDICTION = 'MassCorrectedPrediction_{}.petsc'
PATTERN_SPINUP_PREDICTION = 'Spinup_1000_Prediction_{}.petsc'
PATTERN_SPINUP_MASSCORRECTED_PREDICTION = 'Spinup_1000_MassCorrectedPrediction_{}.petsc'
PATTERN_SPINUP_TOLERANCE_MASSCORRECTED_PREDICTION = 'SpinupTolerance_MassCorrectedPrediction_{}.petsc'
PATTERN_SPINUP_TOLERANCE = 'SpinupTolerance_{}.petsc'

def main():
    """
    Create the plots and tables for the paper.
    @author: Markus Pfeil
    """
    annPlot = ANN_Plot(orientation='gmd', fontsize=8, dbpath=PATH_DATABASE, completeTable=False)
    
    #Parameter for the different plots
    norm = '2'
    showmeans = False
    showquartile = True
    cpus = 64
    rmax = 0.325
    colorloss = 'C7'

    #Plots for the fully connected network
    annId = 0
    color = 'C0'

    #Plot losses of the training (Figure 4.1)
    plotTraining(annPlot, annId, colors=[color, colorloss])

    #Create table of relative difference using different norms (Table 4.1)
    table = tableNormConcentrationDifference(annId)
    print('Table:\n{}'.format(table))

    #Surface plot (Figure 4.2, 4.3 and 4.4)
    plotTracerConcentrationSurface(annId, 0, massAdjustment=False, predictionMetos3d=False, plotSlice=True, slicenum=[117])
    plotTracerConcentrationSurface(annId, 0, massAdjustment=True, predictionMetos3d=False, plotSlice=True, slicenum=[117])
    plotTracerConcentrationSurface(annId, 0, massAdjustment=True, predictionMetos3d=True, plotSlice=True, slicenum=[117], vmax=0.00016)

    #Tatortplots (Figure 4.5: relative error for all sets of model parameter)
    plotScatterNorm(annPlot, annId, orientation='etnatp4', rmax=0.4, color=color)

    #Spinup norm (Figure 4.6)
    plotSpinupData(annPlot, annId, 0, massAdjustment=True, tolerance=10**(-4), color=color)

    #Histogram with required model years (Figure 4.7)
    plotHistSpinupYear(annPlot, annId, massAdjustment=True, tolerance=10**(-4), color=color)


    #Plots for the ANN trained with the SET algorithm
    annId = 1
    color = 'C1'

    #Plot losses of the training (Figure 4.8)
    plotTraining(annPlot, annId, colors=[color, colorloss])

    #Tatortplots (Figure 4.9: relative error for all sets of model parameter)
    plotScatterNorm(annPlot, annId, orientation='etnatp4', rmax=0.4, color=color)

    #Histogram with required model years (Figure 4.10)
    plotHistSpinupYear(annPlot, annId, massAdjustment=True, tolerance=10**(-4), color=color)


    #Plots for the ANN designed with the genetic algorithm
    annId = 2
    color = 'C9'

    #Plot losses of the training (Figure 4.11)
    plotTraining(annPlot, annId, colors=[color, colorloss])

    #Tatortplots (Figure 4.12: relative error for all sets of model parameter)
    plotScatterNorm(annPlot, annId, orientation='etnatp4', rmax=0.4, color=color)

    #Histogram with required model years (Figure 4.13)
    plotHistSpinupYear(annPlot, annId, massAdjustment=True, tolerance=10**(-4), color=color)


    #Violinplot (Figure 5.1)
    plotViolinSpinupYear(annPlot, annIdList=[0, 1, 2], massAdjustment=True, tolerance=10**(-4), colors=['C3', 'C0', 'C1', 'C9'])



def tableNormConcentrationDifference(annId, model='N', parameterId=0):
    """
    Create a string for a table in LaTeX with the relative concentration difference using different norms.
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert model in Metos3d_Constants.METOS3D_MODELS
    assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX_TEST+1)

    annDb = Ann_Database(dbpath=PATH_DATABASE, completeTable=False)
    concentrationId = annDb.get_concentrationId_annValues(annId)

    tableStr = '$\\mathbf{x}$ & $\\left\\| \\cdot \\right\\|_2$ & $\\left\\| \\cdot \\right\\|_{2,V}$ & $\\left\\| \\cdot \\right\\|_{2,T}$ & $\\left\\| \\cdot \\right\\|_{2,V,T}$ \\\\\n'
    tableStr += '\\hline\n'

    combinations = [(False, 0, '$\\mathbf{y}_{\\text{ANN}}$'), (False, 1000, '$\\mathbf{y}^{1000}_{\\text{ANN}}$'), (True, 0, '$\\mathbf{\\tilde{y}}_{\\text{ANN}}$'), (True, 1000, '$\\mathbf{\\tilde{y}}^{1000}_{\\text{ANN}}$')]
    for (massAdjustment, year, label) in combinations:
        tableStr += '{:40s}'.format(label)
        for (norm, trajectory) in [('2', ''), ('Boxweighted', ''), ('2', 'Trajectory'), ('Boxweighted', 'Trajectory')]:
            normValue = annDb.read_rel_norm(model, concentrationId, massCorrection=massAdjustment, year=year, norm=norm, parameterId=parameterId, trajectory=trajectory)
            assert len(normValue) == 1
            tableStr += ' & {:.3e}'.format(normValue[0])
        tableStr += ' \\\\\n'

    annDb.close_connection()

    return tableStr


def plotTraining(annPlot, annId, colors=['C0', 'C3'], orientation='gmd', fontsize=8):
    """
    Plot the losses of the training.
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type(colors) is list and len(colors) == 2

    annPlot._init_plot(orientation=orientation, fontsize=fontsize)
    annPlot.plot_training_loss(annId, colors=colors)
    annPlot.set_subplot_adjust(left=0.145, bottom=0.16, right=0.97, top=0.995)
    filenameSpinup = os.path.join(PATH_FIGURE, 'ANNTraining_AnnId_{:d}.pdf'.format(annId))
    annPlot.savefig(filenameSpinup)
    annPlot.close_fig()
     

def plotScatterNorm(annPlot, annId, orientation='etnatp4', rmax=0.4, color='b'):
    """
    Plot the four scatter plots for the given annId.
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)

    rposition=55

    annPlot.init_subplot(1, 4, orientation=orientation, subplot_kw={'projection': 'polar'})

    annPlot.plot_scatter_rel_norm(annId, massAdjustment=False, tolerance=None, year=0, rmax=rmax, rposition=rposition, color=color)
    annPlot._axesResult.xaxis.grid(False)
    annPlot._axesResult.set_xticklabels([])
    annPlot._axesResult.set_rmax(rmax)
    annPlot._axesResult.set_yticks([0.2, 0.4], minor=False)
    annPlot._axesResult.set_yticks([0.1, 0.3], minor=True)
    annPlot._axesResult.yaxis.grid(True, which='minor', lw=0.5, alpha=0.5)
    annPlot._axesResult.set_rlabel_position(rposition)
    annPlot.set_subplot_adjust(left=0.005, bottom=0.005, right=0.995, top=0.995)

    #The mass corrected prediction
    annPlot.set_subplot(0,1)
    annPlot.plot_scatter_rel_norm(annId, massAdjustment=True, tolerance=None, year=0, rmax=rmax, rposition=rposition, color=color)
    annPlot._axesResult.xaxis.grid(False)
    annPlot._axesResult.set_xticklabels([])
    annPlot._axesResult.set_rmax(rmax)
    annPlot._axesResult.set_yticks([0.2, 0.4], minor=False)
    annPlot._axesResult.set_yticks([0.1, 0.3], minor=True)
    annPlot._axesResult.yaxis.grid(True, which='minor', lw=0.5, alpha=0.5)
    annPlot._axesResult.set_rlabel_position(rposition)
    annPlot.set_subplot_adjust(left=0.005, bottom=0.005, right=0.995, top=0.995)

    #The spin up calculation over 1000 model years
    annPlot.set_subplot(0,2)
    annPlot.plot_scatter_rel_norm(annId, massAdjustment=True, tolerance=None, year=1000, rmax=rmax, rposition=rposition, color=color)
    annPlot._axesResult.xaxis.grid(False)
    annPlot._axesResult.set_xticklabels([])
    annPlot._axesResult.set_yticks([0.2, 0.4], minor=False)
    annPlot._axesResult.set_yticks([0.1, 0.3], minor=True)
    annPlot._axesResult.yaxis.grid(True, which='minor', lw=0.5, alpha=0.5)
    annPlot._axesResult.set_rlabel_position(rposition)
    annPlot.set_subplot_adjust(left=0.005, bottom=0.005, right=0.995, top=0.995)


    #The spin up calculation over 1000 model years
    annPlot.set_subplot(0,3)
    annPlot.plot_scatter_rel_norm(annId, massAdjustment=False, tolerance=None, year=1000, rmax=rmax, rposition=rposition, color=color)
    annPlot._axesResult.xaxis.grid(False)
    annPlot._axesResult.set_xticklabels([])
    annPlot._axesResult.set_yticks([0.2, 0.4], minor=False)
    annPlot._axesResult.set_yticks([0.1, 0.3], minor=True)
    annPlot._axesResult.yaxis.grid(True, which='minor', lw=0.5, alpha=0.5)
    annPlot._axesResult.set_rlabel_position(rposition)
    annPlot.set_subplot_adjust(left=0.005, bottom=0.005, right=0.995, top=0.995)

    plt.tight_layout(pad=0.05, w_pad=0.75)

    filenameScatter = os.path.join(PATH_FIGURE, '{}{}.pdf'.format('NormScatter', mapFilename(annId)))
    annPlot.savefig(filenameScatter)
    annPlot.close_fig()


def plotScatterRelNorm(annPlot, annId, massAdjustment=False, tolerance=None, cpus=64, year=None, norm='2', orientation='etnatp', fontsize=8, alpha=0.75, rmax=None, rposition=55, color='b'):
    """
    Plot the scatter polor plot with the relative norm (between the metos3d solution initilized with the prediction and the reference solution divided by the norm of the reference parameter) for all parameter sets of the latin hypercube sample for the given ann setting.
    Year: None: Use the last year. Otherwise use the given year.
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or tolerance > 0
    assert type(cpus) is int and cpus > 0
    assert year is None or type(year) is int and year >= 0
    assert norm in ['2', 'Boxweighted', 'BoxweightedVol']
    assert type(alpha) is float
    assert rmax is None or type(rmax) is float and 0 < rmax
    assert type(rposition) is int and 0 <= rposition and rposition <= 360
    
    annPlot._init_plot(orientation=orientation, fontsize=fontsize, projection='polar')
    annPlot.plot_scatter_rel_norm(annId, massAdjustment=massAdjustment, tolerance=tolerance, cpus=cpus, year=year, norm=norm, alpha=alpha, rmax=rmax, rposition=rposition, color=color)
    annPlot.set_subplot_adjust(left=0.005, bottom=0.005, right=0.995, top=0.995)
    filenameScatter = os.path.join(PATH_FIGURE, '{}{}.pdf'.format('NormScatter', mapFilename(annId, massAdjustment=massAdjustment, tolerance=tolerance, year=year)))
    annPlot.savefig(filenameScatter)
    annPlot.close_fig()


def plotSpinupData(annPlot, annId, parameterId, massAdjustment=False, tolerance=None, cpus=64, maincolor='C3', color='b', orientation='gmd', fontsize=8):
    """
    Plot the spin up data of the simulation using spin up tolerance
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or tolerance > 0
    assert type(cpus) is int and cpus > 0

    annPlot._init_plot(orientation=orientation, fontsize=fontsize)
    annPlot.plot_spinup_data(annId, parameterId, massAdjustment=massAdjustment, tolerance=tolerance, cpus=cpus, colors=[maincolor, color])
    annPlot.set_subplot_adjust(left=0.1525, bottom=0.165, right=0.995, top=0.995)
    filenameSpinup = os.path.join(PATH_FIGURE, '{}{}.pdf'.format('Spinup', mapFilename(annId, massAdjustment=massAdjustment, tolerance=tolerance)))
    annPlot.savefig(filenameSpinup)
    annPlot.close_fig()


def plotHistSpinupYear(annPlot, annId, massAdjustment=False, tolerance=None, cpus=64, bins=25, orientation='gmd', fontsize=8, maincolor='C3', color='b'):
    """
    Plot the histogram with the required years to reach the given spin up tolerance for all parameter sets of the latin hypercube sample for the given ann setting.
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type (massAdjustment) is bool
    assert tolerance is None or tolerance > 0
    assert type(cpus) is int and cpus > 0
    assert type(bins) is int and bins > 1

    annPlot._init_plot(orientation=orientation, fontsize=fontsize)
    annPlot.plot_hist_spinup_year(annId, massAdjustment=massAdjustment, tolerance=tolerance, cpus=cpus, bins=bins, maincolor=maincolor, color=color)
    annPlot.set_subplot_adjust(left=0.105, bottom=0.165, right=0.995, top=0.84)
    filenameHist = os.path.join(PATH_FIGURE, '{}{}.pdf'.format('SpinupYear_Histogram', mapFilename(annId, massAdjustment, tolerance)))
    annPlot.savefig(filenameHist)
    annPlot.close_fig()


def plotViolinSpinupYear(annPlot, annIdList, massAdjustment=False, tolerance=None, cpus=64, showmeans=False, showquartile=True, colors=None, orientation='gmd', fontsize=8):
    """
    Plot the violin plot with the required years to reach the given spin up tolerance for all parameter sets of the latin hypercube sample for the given anns.
    @author: Markus Pfeil
    """
    assert type(annIdList) is list and len(annIdList) > 0
    assert type(massAdjustment) is bool
    assert tolerance is None or tolerance > 0
    assert cpus > 0
    assert type(showmeans) is bool
    assert type(showquartile) is bool
    assert colors is None or type(colors) is list and len(colors) >= len(annIdList)

    annPlot._init_plot(orientation=orientation, fontsize=fontsize)
    annPlot.plot_violinplot_spinup_avg_year(annIdList, massAdjustment=massAdjustment, tolerance=tolerance, cpus=cpus, points=100, showmeans=showmeans, showquartile=showquartile, colors=colors)
    annPlot.set_subplot_adjust(left=0.145, bottom=0.15, right=0.995, top=0.995)
    filenameViolin = os.path.join(PATH_FIGURE, 'ViolinPlot_SpinupYears_ANN.pdf')
    annPlot.savefig(filenameViolin)
    annPlot.close_fig()


def plotTracerConcentrationSurface(annId, parameterId, massAdjustment=False, tolerance=None, tracerDifference=True, relativeError=True, predictionMetos3d=False, cmap=None, plotSurface=True, plotSlice=False, slicenum=None, orientation='etnasp', vmin=0.0, vmax=0.002):
    """
    Plot the tracer concentration for a given layer
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, 3)
    assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
    assert slicenum is None or (slicenum is not None and isinstance(slicenum, list))
    assert type(massAdjustment) is bool
    assert tolerance is None or (tolerance > 0 and massAdjustment)

    annDb = Ann_Database(dbpath=PATH_DATABASE, completeTable=False)
    annTyp, annNumber, model = annDb.get_annTypeNumberModel(annId)
    annDb.close_connection()

    #Prediction of the tracer concentration using the neural network
    Directory_Dict = {0: 'FCN', 1: 'SET', 2: 'GA'}
    simulationPath = os.path.join(PATH_ANN_RESULTS, Directory_Dict[annId], 'Parameter_{:0>3d}'.format(parameterId))

    if not predictionMetos3d:
        filename_prediction = os.path.join(simulationPath, PATTERN_PREDICTION if not massAdjustment else PATTERN_MASSCORRECTED_PREDICTION)
    else:
        if tolerance is None:
            filename_prediction = os.path.join(simulationPath, PATTERN_SPINUP_PREDICTION if not massAdjustment else PATTERN_SPINUP_MASSCORRECTED_PREDICTION)
        else:
            filename_prediction = os.path.join(simulationPath, PATTERN_SPINUP_TOLERANCE_MASSCORRECTED_PREDICTION)

    file_prediction_exists = True
    for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
        file_prediction_exists = file_prediction_exists and os.path.exists(filename_prediction.format(trac)) and os.path.isfile(filename_prediction.format(trac))

    if file_prediction_exists:
        tracerConcentration = np.zeros(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[model])))
        i = 0
        for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
            tracerConcentration[:, i] = petsc.readPetscFile(filename_prediction.format(trac))
            i = i + 1
    else:
        #There is no file with the tracer concentration
        assert False

    #Calculate the norm of the tracer concentration vector
    if relativeError:
        tracerVec = np.zeros(shape=(Metos3d_Constants.METOS3D_VECTOR_LEN, len(Metos3d_Constants.METOS3D_MODEL_TRACER[model])))
        i = 0
        for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
            filename_tracer = os.path.join(PATH_REFERENCE_RESULTS, 'Parameter_{:0>3d}'.format(parameterId), PATTERN_SPINUP_TOLERANCE.format(trac))
            tracerVec[:,i] = petsc.readPetscFile(filename_tracer)
            i = i + 1
        normValue = np.linalg.norm(tracerVec)
    else:
        normValue = 1.0

    #Plot the tracer concentration for every tracer
    i = 0
    for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[model]:
        #Set path to the tracer file
        filename_tracer = os.path.join(PATH_REFERENCE_RESULTS, 'Parameter_{:0>3d}'.format(parameterId), PATTERN_SPINUP_TOLERANCE.format(tracer))
        assert os.path.exists(filename_tracer) and os.path.isfile(filename_tracer)
        #Read tracer vector
        tracerVecMetos3d = petsc.readPetscFile(filename_tracer)

        #Calculate the difference between the prediction of the neural network and the concentration computed with metos3d and time step 1dt
        if tracerDifference:
            v1d = np.divide(np.fabs(tracerConcentration[:,i] - tracerVecMetos3d), normValue)
        else:
            v1d = np.divide(np.fabs(tracerConcentration[:,i]), normValue)

        surface = SurfacePlot(orientation=orientation)
        surface.init_subplot(1, 2, orientation=orientation, gridspec_kw={'width_ratios': [9,5]})

        #Plot the surface concentration
        meridians = None if slicenum is None else [np.mod(Metos3d_Constants.METOS3D_GRID_LONGITUDE * x, 360) for x in slicenum]
        cntr = surface.plot_surface(v1d, projection='robin', levels=50, vmin=vmin, vmax=vmax, ticks=plt.LinearLocator(6), format='%.1e', pad=0.05, extend='max', meridians=meridians, colorbar=False)

        #Plot the slice plan of the concentration
        if plotSlice and slicenum is not None:
            surface.set_subplot(0,1)
            for s in slicenum:
                surface.plot_slice(v1d, s, levels=50, vmin=vmin, vmax=vmax, ticks=plt.LinearLocator(6), format='%.1e', pad=0.02, extend='max', colorbar=False)

        plt.tight_layout(pad=0.05, w_pad=0.15)
        cbar = surface._fig.colorbar(cntr, ax=surface._axes[0], format='%.1e', ticks=plt.LinearLocator(5), pad=0.02, aspect=40, extend='max', orientation='horizontal', shrink=0.8)

        filename_surface = os.path.join(PATH_FIGURE, '{}{}.pdf'.format('Surface', mapFilename(annId, massAdjustment, year=1000 if predictionMetos3d else 0)))
        surface.savefig(filename_surface)

        i = i + 1


def mapFilename(annId, massAdjustment=False, tolerance=None, year=None):
    """
    Map the parameter to the postfix of the filename
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or tolerance > 0
    assert year is None or type(year) is int and year >= 0

    filename = ''

    #Add part for the used ANN    
    if annId == 0:
        filename = filename + '_FCN'
    elif annId == 1:
        filename = filename + '_SET'
    elif annId == 2:
        filename = filename + '_GEN'
    else:
        assert False

    if not massAdjustment:
        filename = filename + '_Prediction'
    elif tolerance is None and year == 0:
        filename = filename + '_AdjustedPrediction'
    elif tolerance is None and year == 1000:
        filename = filename + '_Metos3dWithAdjustedPrediction'
    elif tolerance == 10**(-4) and year is None:
        filename = filename + '_Metos3dWithAdjustedPredictionSpinup'
    else:
        assert False

    return filename


if __name__ == '__main__':
    main()

