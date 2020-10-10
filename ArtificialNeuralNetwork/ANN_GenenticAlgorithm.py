#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import argparse
import logging

import ANN_Config_GA
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.ANN_GeneticAlgorithm import ANN_GeneticAlgorithm


def main(gid):
    """

    @author: Markus Pfeil
    """
    assert gid is None or int(gid) >= 0

    #Parse config for the given gid
    if gid is None:
        config = ANN_Config_GA.ANN_Config_Default
    else:
        gid = int(gid)
        config = parseConfig(gid)

    #Set log file
    pathLogs = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid), 'Logs')
    os.makedirs(pathLogs, exist_ok=True)
    filenameLogs = os.path.join(pathLogs, GA_Constants.PATTERN_LOGFILE_GENETIC_ALGORITHM.format(gid))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=filenameLogs, filemode='a', level=logging.DEBUG)

    ga = ANN_GeneticAlgorithm(gid=gid, algorithm=config['algorithm'], generations=config['generations'], populationSize=config['populationSize'], metos3dModel=config['metos3dModel'])
    ga.run(config=config['config'], gaParameter=config['gaParameter'])


def parseConfig(gid):
    """
    Parse the config for the given gid.
    @author: Markus Pfeil
    """
    assert type(gid) is int and gid >= 0

    config = {}

    #Use default config if no value is specified for the parameter
    for key in ANN_Config_GA.ANN_Config_Default:
        try:
            config[key] = ANN_Config_GA.ANN_Config_GA[gid][key]
        except KeyError:
            config[key] = ANN_Config_GA.ANN_Config_Default[key]

    return config


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("-gid", nargs='?', const=None, default=None, help="Id of the run using the genetic algorithm.")

    args = parser.parse_args()

    main(gid=args.gid)

