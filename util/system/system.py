#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import neshCluster.constants as NeshCluster_Constants
import standaloneComputer.constants as PC_Constants

try:
    USER = os.environ['USER']
except KeyError:
    assert (False, 'Not possible to determine the used system on the basis of the system variable USER')
else:
    if USER == 'sunip350':
        #NEC HPC system of the CAU Kiel
        SYSTEM = 'NEC-NQSV'
        DATA_PATH = NeshCluster_Constants.DATA_PATH
        PYTHON_PATH = NeshCluster_Constants.PYTHON_PATH
        FIGURE_PATH = NeshCluster_Constants.FIGURE_PATH
        BACKUP_PATH = NeshCluster_Constants.BACKUP_PATH
        METOS3D_MODEL_PATH = DATA_PATH
    elif USER == 'mpf':
        #Standalone computer of the User mpf at the CAU Kiel
        SYSTEM = 'PC'
        DATA_PATH = PC_Constants.DATA_PATH
        PYTHON_PATH = PC_Constants.PYTHON_PATH
        FIGURE_PATH = PC_Constants.FIGURE_PATH
        BACKUP_PATH = ''
        METOS3D_MODEL_PATH = DATA_PATH 
    else:
        assert (False, 'There are not paths defined for the given system using the user {}'.format(USER))

