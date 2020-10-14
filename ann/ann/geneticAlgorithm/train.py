#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import gc
import random
from keras import backend as K

import metos3dutil.metos3d.constants as Metos3d_Constants
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
from ann.network.ANN_SET_MLP import SET_MLP


def train_and_score(genome, metos3dModel, gid, indexMin=ANN_Constants.PARAMETERID_MAX_TEST+1, indexMax=ANN_Constants.PARAMETERID_MAX, trainingSize=None):
    """
    Train the model, return test loss.
    @author: Markus Pfeil
    """
    assert type(genome.geneparam) is dict
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(gid) is int and gid >= 0
    assert type(indexMin) is int and 0 <= indexMin
    assert type(indexMax) is int and indexMin < indexMax and indexMax <= ANN_Constants.PARAMETERID_MAX
    assert trainingSize is None or type(trainingSize) is int and 0 < trainingSize and trainingSize <= indexMax - indexMin

    mlp = SET_MLP(genome.u_ID, metos3dModel=metos3dModel, setPath=False)
    assert genome.geneparam['nb_layers'] == len(genome.geneparam['nb_neurons'])
    mlp.create_model(unitsPerLayer=genome.geneparam['nb_neurons'], activation=[genome.geneparam['activation'] for i in range(genome.geneparam['nb_layers'] + 1)]) # +1 for the output layer

    #Set path for the neural network trained in in the genetic algorithm
    ann_path = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.generation, genome.u_ID))
    mlp.set_path(ann_path)

    #Set model parameter
    if trainingSize is not None:
        mlp.set_dataIndicesList(random.sample(range(indexMin, indexMax), trainingSize))
    else:
        mlp.set_dataIndices(indexMin=indexMin, indexMax=indexMax)
    mlp.set_maxepoches(genome.geneparam['maxepoches'])
    mlp.set_SETepoches(genome.geneparam['SETepoches'])
    mlp.set_epsilon(float(genome.geneparam['epsilon']))
    mlp.set_zeta(genome.geneparam['zeta'])
    mlp.set_learning_rate(genome.geneparam['learning_rate'])
    try:
        mlp.set_conserveMass(genome.geneparam['conserveMass'])
    except KeyError:
        pass

    mlp.train()
	
    #Save the neural network and the training monitor
    mlp.saveAnn()
    mlp.saveTrainingMonitoring()
    K.clear_session()
    gc.collect()

    return mlp.getLoss()

