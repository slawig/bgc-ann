#!/usr/bin/env python
# -*- coding: utf8 -*

import gc
import argparse
import random

import ANN_Config_FCN
import ANN_Config_SET
from ann.network.FCN import FCN
from ann.network.ANN_SET_MLP import SET_MLP


def train(annType, annNumber):
    """
    Generate and train the neural network for the given annNumber using a FCN or the SET algorithm.
    @author: Markus Pfeil
    """
    assert annType in ['fcn', 'set']
    assert type(annNumber) is int and 0 <= annNumber
    
    config = parseConfig(annType, annNumber)

    if annType == 'fcn':
        assert type(annNumber) is int and annNumber in ANN_Config_FCN.ANN_Config_FCN
        mlp = FCN(annNumber)
    elif annType == 'set':
        assert type(annNumber) is int and annNumber in ANN_Config_SET.ANN_Config_SET
        mlp = SET_MLP(annNumber)
        mlp.set_conserveMass(config["ConserveMass"])
    else:
        assert False

    if type(config["TrainingsData"]) is list and len(config["TrainingsData"]) == 2:
        assert config["TrainingsData"][0] <= config["TrainingsData"][1]
        mlp.set_dataIndices(indexMin=config["TrainingsData"][0], indexMax=config["TrainingsData"][1])
    elif type(config["TrainingsData"]) is tuple and len(config["TrainingsData"]) == 3:
        assert config["TrainingsData"][0] <= config["TrainingsData"][1] and config["TrainingsData"][2] <= config["TrainingsData"][1] - config["TrainingsData"][0]
        mlp.set_dataIndicesList(random.sample(range(config["TrainingsData"][0], config["TrainingsData"][1]), config["TrainingsData"][2]))
    mlp.set_maxepoches(config["Epoches"])
    mlp.create_model(unitsPerLayer=config["Layer"], activation=config["Activation"])
    mlp.train()
    mlp.saveAnn()
    mlp.saveTrainingMonitoring()

    gc.collect()


def parseConfig(annType, annNumber):
    """
    Parse the config for the given annNumber.
    @author: Markus Pfeil
    """
    assert annType in ['fcn', 'set']
    
    config = {}

    if annType == 'fcn':
        assert type(annNumber) is int and annNumber in ANN_Config_FCN.ANN_Config_FCN

        #Use default config if no value is specified for the parameter
        for key in ANN_Config_FCN.ANN_Config_Default:
            try:
                config[key] = ANN_Config_FCN.ANN_Config_FCN[annNumber][key]
            except KeyError:
                config[key] = ANN_Config_FCN.ANN_Config_Default[key]
    elif annType == 'set':
        assert type(annNumber) is int and annNumber in ANN_Config_SET.ANN_Config_SET

        #Use default config if no value is specified for the parameter
        for key in ANN_Config_SET.ANN_Config_Default:
            try:
                config[key] = ANN_Config_SET.ANN_Config_SET[annNumber][key]
            except KeyError:
                config[key] = ANN_Config_SET.ANN_Config_Default[key]
    else:
        assert False
 
    return config 


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("annType", type=str, help="Type of the ann (fcn or set).")
    parser.add_argument("annNumber", type=int, help="annNumber of the neural network using a FCN or the SET algorithm.")
    
    args = parser.parse_args()

    assert args.annType in ['fcn', 'set']
    assert type(args.annNumber) is int and 0 <= args.annNumber
    train(args.annType, args.annNumber)

