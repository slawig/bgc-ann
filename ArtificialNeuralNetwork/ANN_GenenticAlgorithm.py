#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import argparse
import logging

import ANN_Config_GA
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.geneticAlgorithm import GeneticAlgorithm
from ann.geneticAlgorithm.geneticAlgorithmRechenberg import GeneticAlgorithm as GeneticAlgorithmRechenberg


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

    if config['algorithm'] == 'GeneticAlgorithm':
        if len(config['gaParameter']) == 3:
            assert 'retain' in config['gaParameter'] and type(config['gaParameter']['retain']) is float and 0.0 < config['gaParameter']['retain'] and config['gaParameter']['retain'] < 1.0
            assert 'random_select' in config['gaParameter'] and type(config['gaParameter']['random_select']) is float and 0.0 < config['gaParameter']['random_select'] and config['gaParameter']['random_select'] < 1.0
            assert 'mutate_chance' in config['gaParameter'] and type(config['gaParameter']['mutate_chance']) is float and 0.0 < config['gaParameter']['mutate_chance'] and config['gaParameter']['mutate_chance'] < 1.0

            evolverParameter = config['gaParameter']
        else:
            evolverParameter = GA_Constants.GA_Config_GeneticAlgorithm_Default

        logging.info('***Use probabilities for genetic algorithm: retain {:f} random_select {:f} and mutate_change {:f}***'.format(evolverParameter['retain'], evolverParameter['random_select'], evolverParameter['mutate_chance']))
        ga = GeneticAlgorithm(gid=gid, populationSize=config['populationSize'], generations=config['generations'], metos3dModel=config['metos3dModel'], retain=evolverParameter['retain'], random_select=evolverParameter['random_select'], mutate_chance=evolverParameter['mutate_chance'])

    elif config['algorithm'] == 'Rechenberg':
        if len(config['gaParameter']) == 2:
            assert 'alpha' in config['gaParameter'] and type(config['gaParameter']['alpha']) is float and 0.0 < config['gaParameter']['alpha']
            assert 'offspring' in config['gaParameter'] and type(config['gaParameter']['offspring']) is int and 0 < config['gaParameter']['offspring']

            evolverParameter = config['gaParameter']
        else:
            evolverParameter = GA_Constants.GA_Config_Rechenberg_Default

        logging.info('***Used parameter for evolutionary strategy (Rechenberg): alpha {:f} and offspring {:d}***'.format(evolverParameter['alpha'], evolverParameter['offspring']))

        ga = GeneticAlgorithmRechenberg(gid=gid, populationSize=config['populationSize'], generations=config['generations'], metos3dModel=config['metos3dModel'], alpha=evolverParameter['alpha'], offspring=evolverParameter['offspring'])
    else:
        assert False, 'Not implemented genetic algorithm {:s}'.format(self._algorithm)

    ga.run(config=config['config'])


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

