#!/usr/bin/env python
# -*- coding: utf8 -*

import ann.network.constants as ANN_Constants


ANN_Config_Default = {"Layer": [10, 25, 211],
                      "Activation": ['elu', 'elu', 'elu', 'relu'],
                      "ConserveMass": False,
                      "TrainingsData": [ANN_Constants.PARAMETERID_MAX_TEST+1, ANN_Constants.PARAMETERID_MAX],
                      "Epoches": 1000,
                      "ValidationSplit": None,
                      "KfoldSplit": None}


#ANN_Config_SET includes the configuration for the trained neural networks using the SET algorithm.
#ANN_Config_SET defines only the differences to the default configuraiton ANN_Config_Default.
ANN_Config_SET = {}

ANN_Config_SET[0] = {"TrainingsData": [0, 100]}
ANN_Config_SET[1] = {"TrainingsData": [0, 1100]}
ANN_Config_SET[2] = {"TrainingsData": [0, 1100]}
ANN_Config_SET[3] = {"Layer": [10, 100, 1000],
                     "TrainingsData": [0, 1100]}
ANN_Config_SET[4] = {"Layer": [10, 100, 500, 1000],
                     "Activation": ['elu', 'elu', 'elu', 'elu', 'relu'], 
                     "TrainingsData": [0, 1100]}
ANN_Config_SET[5] = {"Layer": [20, 50, 300]}
ANN_Config_SET[6] = {"TrainingsData": [0, 11100]}
ANN_Config_SET[7] = {"ConserveMass": True}
ANN_Config_SET[8] = {"ConserveMass": False}
ANN_Config_SET[9] = {"ConserveMass": True}
ANN_Config_SET[10] = {"ConserveMass": True}
ANN_Config_SET[11] = {"TrainingsData": [0, 100]}
ANN_Config_SET[12] = {"TrainingsData": (101, 1100, 100)}
ANN_Config_SET[13] = {"TrainingsData": (101, 1100, 100)}
ANN_Config_SET[14] = {"TrainingsData": (101, 1100, 100)}
ANN_Config_SET[15] = {"TrainingsData": (101, 1100, 100)}
