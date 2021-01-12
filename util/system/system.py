#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import neshCluster.constants as NeshCluster_Constants
import standaloneComputer.constants as PC_Constants

BATCH_SYSTEM_ENV_NAME = 'BATCH_SYSTEM'

try:
    BATCH_SYSTEM_STR = os.environ[BATCH_SYSTEM_ENV_NAME]
except KeyError:
    SYSTEM = 'PC'
    DATA_PATH = PC_Constants.DATA_PATH
    PYTHON_PATH = PC_Constants.PYTHON_PATH
    FIGURE_PATH = PC_Constants.FIGURE_PATH
    BACKUP_PATH = ''
    METOS3D_MODEL_PATH = DATA_PATH 
else:
    SYSTEM = 'NEC-NQSV'
    DATA_PATH = NeshCluster_Constants.DATA_PATH
    PYTHON_PATH = NeshCluster_Constants.PYTHON_PATH
    FIGURE_PATH = NeshCluster_Constants.FIGURE_PATH
    BACKUP_PATH = NeshCluster_Constants.BACKUP_PATH
    METOS3D_MODEL_PATH = DATA_PATH

