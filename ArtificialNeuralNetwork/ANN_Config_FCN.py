#!/usr/bin/env python
# -*- coding: utf8 -*

import ann.network.constants as ANN_Constants


ANN_Config_Default = {"Layer": [10, 25, 211],
                      "Activation": ['elu', 'elu', 'elu', 'selu'],
                      "TrainingsData": [ANN_Constants.PARAMETERID_MAX_TEST+1, ANN_Constants.PARAMETERID_MAX],
                      "Epoches": 1000,
                      "ValidationSplit": None,
                      "KfoldSplit": None}


#ANN_Config_FCN includes the configuration for the trained neural networks.
#ANN_Config_FCN defines only the differences to the default configuraiton ANN_Config_Default.
ANN_Config_FCN = {}

for i in range(50):
    ANN_Config_FCN[i] = {"TrainingsData": [0, 33]}
    ANN_Config_FCN[50+i] = {"TrainingsData": [0, 66],
                            "ValidationSplit": 0.25}

ANN_Config_FCN[100] = {"TrainingsData": [0, 100]}

ANN_Config_FCN[101] = {"TrainingsData": [0, 100],
                       "Epoches": 20000}

ANN_Config_FCN[102] = {"TrainingsData": [0, 100],
                       "KfoldSplit": 5}

ANN_Config_FCN[103] = {"Activation": ['elu', 'elu', 'elu', 'relu'],
                       "TrainingsData": [0, 1100]}

ANN_Config_FCN[104] = {"Activation": ['elu', 'elu', 'elu', 'relu']}

ANN_Config_FCN[105] = {"TrainingsData": (101, 1100, 100)}
ANN_Config_FCN[106] = {"TrainingsData": (101, 1100, 100)}
ANN_Config_FCN[107] = {"Activation": ['elu', 'elu', 'elu', 'relu'],
                       "TrainingsData": (101, 1100, 100)}
ANN_Config_FCN[108] = {"Activation": ['elu', 'elu', 'elu', 'relu'],
                       "TrainingsData": (101, 1100, 100)}
