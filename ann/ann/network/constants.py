#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
from system.system import DATA_PATH, FIGURE_PATH


PATH = os.path.join(DATA_PATH, 'ArtificialNeuralNetwork')
PATH_LHS = os.path.join(DATA_PATH, 'LatinHypercubeSample')
PATH_TRACER = os.path.join(PATH, 'Tracer')
PATH_FIGURE = FIGURE_PATH


# Path of the neural networks using the set algorithm
PATH_SET = os.path.join(PATH, 'FCN_SET')
PATH_FCN = os.path.join(PATH, 'FCN')
FCN = 'FCN_{:0>3d}'
FCN_SET = 'FCN_SET_{:0>3d}'

# ANN parameter
ANN_LAYER_NAME = 'Layer_{:d}'
ANN_FILENAME_FCN = 'ANN_{:0>3d}.h5'
ANN_FILENAME_SET_WEIGHTS = 'ANN_{:0>3d}_weights.h5'
ANN_FILENAME_SET_ARCHITECTURE = 'ANN_{:0>3d}_architecture.json'
ANN_FILENAME_TRAINING_MONITOR = 'ANN_{:0>3d}_training_monitor.txt'

ANN_GENETIC_ALGORITHM_BEST = 'ANN_{:0>3d}_best.txt'

SOLUTIONID_MAX = 2
PARAMETERID_MAX = LHS_Constants.PARAMETERID_MAX
PARAMETERID_MAX_TEST = 100
ANNID_MAX = 249

#Pattern for figures
PATTERN_FIGURE_SPINUP = 'Spinup.{:s}.AnnId_{:0>5d}.ParameterId_{:d}.MassAdjustment_{}.Tolerance_{}.pdf'
PATTERN_FIGURE_HIST_RELNORM = 'Norm.Histogram.{:s}.AnnId_{:0>5d}.MassAdjustment_{}.Tolerance_{}.Year_{}.{}Norm.pdf'
PATTERN_FIGURE_SCATTER_RELNORM = 'Norm.Scatter.{:s}.AnnId_{:0>5d}.MassAdjustment_{}.Tolerance_{}.Year_{}.{}Norm.pdf'
PATTERN_FIGURE_HIST_MASS = 'Mass.Histogram.{:s}.AnnId_{:0>5d}.MassAdjustment_{}.Tolerance_{}.Year_{}.pdf'
PATTERN_FIGURE_HIST_SPINUPTOLERANCE = 'SpinupTolerance.Histogram.{:s}.AnnId_{:0>5d}.MassAdjustment_{}.Tolerance_{}.Year_{}.pdf'
PATTERN_FIGURE_HIST_SPINUPYEAR = 'SpinupYear.Histogram.{:s}.AnnId_{:0>5d}.MassAdjustment_{}.Tolerance_{}.pdf'
PATTERN_FIGURE_HIST_RELSPINUPYEAR = 'RelSpinupYear.Histogram.{:s}.AnnId_{:0>5d}.MassAdjustment_{}.Tolerance_{}.pdf'
PATTERN_FIGURE_VIOLIN_SPINUPYEAR = 'SpinupYear.Violin.{:s}.AnnIds_{}.MassAdjustment_{}.Tolerance_{}.pdf'
PATTERN_FIGURE_SURFACE = 'Surface.{:s}.AnnId_{:d}.ParameterId_{:d}.MassAdjustment_{}.Tolerance_{}.{:s}.relError_{}.Diff_{}.Metos3d_{}.pdf'
PATTERN_FIGURE_SLICE = 'Slice.{:d}.{:s}.AnnId_{:d}.ParameterId_{:d}.MassAdjustment_{}.Tolerance_{}.{:s}.relError_{}.Diff_{}.Metos3d_{}.pdf'

