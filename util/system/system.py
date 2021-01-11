#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import neshCluster.constants as NeshCluster_Constants

BATCH_SYSTEM_ENV_NAME = 'BATCH_SYSTEM'

try:
    BATCH_SYSTEM_STR = os.environ[BATCH_SYSTEM_ENV_NAME]
except KeyError:
    SYSTEM = 'LOGO'
    DATA_PATH = os.path.join('/home', 'mpf', 'Dokumente', 'metos3d')
    PYTHON_PATH = os.path.join('/home', 'mpf', 'Dokumente', 'Python', 'bgc-ann')
    FIGURE_PATH = os.path.join('/home', 'mpf', 'Dokumente', 'Figures')
    BACKUP_PATH = ''
    METOS3D_MODEL_PATH = DATA_PATH 
else:
    SYSTEM = 'NEC-NQSV'
    PYTHON_PATH = NeshCluster_Constants.PYTHON_PATH
    DATA_PATH = NeshCluster_Constants.DATA_PATH
    FIGURE_PATH = NeshCluster_Constants.FIGURE_PATH
    BACKUP_PATH = NeshCluster_Constants.BACKUP_PATH
    METOS3D_MODEL_PATH = DATA_PATH

