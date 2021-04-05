#!/usr/bin/env python
# -*- coding: utf8 -*

import os
from system.system import DATA_PATH, PYTHON_PATH, BACKUP_PATH, FIGURE_PATH

PATH = os.path.join(DATA_PATH, 'InitialValue')
PROGRAM_PATH = os.path.join(PYTHON_PATH, 'InitialValueSimulation')
PATH_FIGURE = os.path.join(FIGURE_PATH, 'InitialValue')
PATH_INITIAL_TRACER = os.path.join(DATA_PATH, 'InitialTracer')

DB_PATH = os.path.join(PATH, 'Database', 'InitialValue_Database.db')
PATH_CONCENTRATION_TYP = {'constant': 'Constant_InitialValue', 'vector': 'Vector_InitialValue'}
PATH_DIFFERENT_TRACER = {1: 'OneTracer', 2: 'TwoTracer', 3: 'ThreeTracer', 4: 'FourTracer', 5: 'FiveTracer'}
PATH_DISTRIBUTION = '{:s}Distribution'
PATH_TRACER_DISTRIBUTION = '{:s}_distribution'

DEFAULT_PYTHONPATH = os.path.join(PYTHON_PATH, 'util') + ':' + os.path.join(PYTHON_PATH, 'initialValue')


PARAMETERID_MAX = 100

PATTERN_JOBFILE = 'Jobfile.{:s}.ParameterId_{:0>3d}.Concentration_Typ_{:s}_Num_{:d}_DifferentTracer_{:d}.Timestep_{:d}dt.txt'
PATTERN_LOGFILE = 'Logfile.{:s}.ParameterId_{:0>3d}.Concentration_Typ_{:s}_Num_{:d}_DifferentTracer_{:d}.Timestep_{:d}dt.log'
PATTERN_JOBOUTPUT = 'Joboutput.{:s}.ParameterId_{:0>3d}.Concentration_Typ_{:s}_Num_{:d}_DifferentTracer_{:d}.Timestep_{:d}dt.out'

PATTERN_TRACER_INITIAL_CONCENTRATION = 'InitialValue_Tracer_{:d}_{:0>3d}.petsc'

