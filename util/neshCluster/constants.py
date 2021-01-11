#!/usr/bin/env python
# -*- coding: utf8 -*

import os

# Stettings for the nesh linux cluster of the CAU Kiel
CPUNUM = {'clmedium': 32, 'cllong': 32, 'clbigmem': 32, 'clexpress': 32}
ELAPSTIM = {'clmedium': 48, 'cllong': 100, 'clbigmem': 200, 'clexpress': 2}
QUEUE = [key for key in CPUNUM]

PARALLEL_JOBS = 20
TIME_SLEEP = 120

#Default parameter
DATA_PATH = os.path.join('/sfs', 'fs2', 'work-sh1', 'sunip350', 'metos3d')
PYTHON_PATH = os.path.join('/sfs', 'fs5', 'home-sh', 'sunip350', 'Python', 'bgc-ann')
PROGRAMM_PATH = os.path.join(PYTHON_PATH, 'ArtificialNeuralNetwork')
FIGURE_PATH = os.path.join('/sfs', 'fs5', 'home-sh', 'sunip350', 'Daten', 'Figures')
BACKUP_PATH = os.path.join('/nfs', 'tape_cache', 'sunip350', 'Daten', 'metos3d')

DEFAULT_QUEUE = 'clmedium'
DEFAULT_CORES = 2
DEFAULT_MEMORY = 20
DEFAULT_PYTHONPATH = os.path.join(PYTHON_PATH, 'util') + ':' + os.path.join(PYTHON_PATH, 'ann') + ':' + os.path.join(PYTHON_PATH, 'sbo') + ':' + PROGRAMM_PATH + ':' + os.path.join(PYTHON_PATH, 'SurrogateBasedOptimization')

