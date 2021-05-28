#!/usr/bin/env python
# -*- coding: utf8 -*

import os
from system.system import DATA_PATH, PYTHON_PATH, BACKUP_PATH, FIGURE_PATH

PATH = os.path.join(DATA_PATH, 'LatinHypercubeSample')
PROGRAM_PATH = os.path.join(PYTHON_PATH, 'TimestepsSimulation')
PATH_FIGURE = os.path.join(FIGURE_PATH, 'Timesteps')

DB_PATH = os.path.join(PATH, 'Database', 'Timesteps_Database.db')

DEFAULT_PYTHONPATH = os.path.join(PYTHON_PATH, 'util') + ':' + os.path.join(PYTHON_PATH, 'timesteps')

PARAMETERID_MAX = 100

CONCENTRATIONID_DICT = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}

PATTERN_JOBFILE = 'Jobfile.{:s}.ParameterId_{:0>3d}.Timestep_{:d}dt.ConcentrationId_{:d}.txt'
PATTERN_LOGFILE = 'Logfile.{:s}.ParameterId_{:0>3d}.Timestep_{:d}dt.ConcentrationId_{:d}.log'
PATTERN_JOBOUTPUT = 'Joboutput.{:s}.ParameterId_{:0>3d}.Timestep_{:d}dt.ConcentrationId_{:d}.out'

#Pattern for figure filenames
PATTERN_FIGURE_SPINUP = 'Spinup.{:s}.ParameterId_{:0>3d}.pdf'
PATTERN_FIGURE_NORM = '{:s}{:s}Norm.{:s}.ParameterId_{:0>3d}.pdf'
PATTERN_FIGURE_SPINUP_NORM = 'ScatterPlot.SpinupNorm_{:s}{:s}.{:s}.pdf'
PATTERN_FIGURE_ERROR_REDUCTION = 'ScatterPlot.ErrorReduction.Norm_{:s}.{:s}.pdf'
PATTERN_FIGURE_REQUIRED_MODEL_YEARS = 'ScatterPlot.RequiredModelYears.Norm_{:s}.{:s}.pdf'
PATTERN_FIGURE_COSTFUNCTION = 'ScatterPlot.Costfunction_{:s}.{:s}.pdf'
PATTERN_FIGURE_OSCILLATION_PARAMETER = 'ScatterPlot.OscillationParameter.{:s}.{:d}dt.pdf'
PATTERN_FIGURE_SURFACE = 'Surface.Timesteps.{:s}.{:d}dt.ParameterId_{:d}.{:s}.relError_{}.diff_{}.pdf'

