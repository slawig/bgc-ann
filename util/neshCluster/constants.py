#!/usr/bin/env python
# -*- coding: utf8 -*

# Stettings for the nesh linux cluster of the CAU Kiel
CPUNUM = {'clmedium': 32, 'cllong': 32, 'clbigmem': 32, 'clexpress': 32}
ELAPSTIM = {'clmedium': 48, 'cllong': 100, 'clbigmem': 200, 'clexpress': 2}
QUEUE = [key for key in CPUNUM]

PARALLEL_JOBS = 20
TIME_SLEEP = 120

#Default parameter
DEFAULT_QUEUE = 'clmedium'
DEFAULT_CORES = 2
DEFAULT_MEMORY = 20
DEFAULT_PYTHONPATH = '/sfs/fs5/home-sh/sunip350/Python/bgc-ann/util:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/ann:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/ArtificialNeuralNetwork'
PROGRAMM_PATH = '/sfs/fs5/home-sh/sunip350/Python/bgc-ann/ArtificialNeuralNetwork'
PYTHON_PATH = '/sfs/fs5/home-sh/sunip350/Python/bgc-ann'

