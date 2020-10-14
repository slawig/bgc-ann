#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import logging
import os
import pickle

import metos3dutil.metos3d.constants as Metos3d_Constants
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.ANN_GeneticAlgorithm import ANN_GeneticAlgorithm
import ann.geneticAlgorithm.genome


def main(filename, metos3dModel='N', gid=0, indexMin=ANN_Constants.PARAMETERID_MAX_TEST+1, indexMax=ANN_Constants.PARAMETERID_MAX, trainingSize=None):
    """
    Train the neural network defined by the given genome, which is stored on the disk.
    @author: Markus Pfeil
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(gid) is int and 0 <= gid
    assert type(indexMin) is int
    assert type(indexMax) is int
    assert trainingSize is None or type(trainingSize) is int

    #Read genome from disk
    genome = None
    genomeFile = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid), filename)
    assert os.path.exists(genomeFile) and os.path.isfile(genomeFile)
    with open(genomeFile, 'rb') as f:
        genome = pickle.load(f)
    
    assert genome is not None
    
    pathLogs = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid), 'Logs', 'LogfileTraining')
    os.makedirs(pathLogs, exist_ok=True)
    logFilename = os.path.join(pathLogs, GA_Constants.PATTERN_LOGFILE_TRANING_GENOME.format(gid, genome.getGeneration(), genome.getUId()))
    logging.basicConfig(filename=logFilename, filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logging.info('***Genetic algorithm {:d}: Training genome with u_id {:d}***'.format(gid, genome.getUId()))
    genome.train(metos3dModel, gid, indexMin=indexMin, indexMax=indexMax, trainingSize=trainingSize)
    
    genomeFilename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.getGeneration(), genome.getUId()), GA_Constants.GENOME_FILENAME.format(gid, genome.getGeneration(), genome.getUId()))

    geneticAlgorithm = ANN_GeneticAlgorithm(gid=gid)
    geneticAlgorithm.saveGenome(genomeFilename, genome)


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename of the stored genome")
    parser.add_argument("metos3dModel", type=str, help="Metos3d model")
    parser.add_argument("gid", type=int, help="Id of the genetic algorithm")
    parser.add_argument("-indexMin", nargs='?', type=int, const=ANN_Constants.PARAMETERID_MAX_TEST+1, default=ANN_Constants.PARAMETERID_MAX_TEST+1, help="Minimal index of the training parameter")
    parser.add_argument("-indexMax", nargs='?', type=int, const=ANN_Constants.PARAMETERID_MAX, default=ANN_Constants.PARAMETERID_MAX, help="Maximal index of the training parameter")
    parser.add_argument("-trainingSize", nargs='?', const=None, default=None, help="Insert the size of the training set")

    args = parser.parse_args()
    trainingSize = None if args.trainingSize is None else int(args.trainingSize)

    main(args.filename, metos3dModel=args.metos3dModel, gid=args.gid, indexMin=args.indexMin, indexMax=args.indexMax, trainingSize=trainingSize)
