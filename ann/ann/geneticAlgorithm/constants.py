#!/usr/bin/env python
# -*- coding: utf8 -*

all_possible_genes = {'nb_layers': [1, 10],
                      'nb_neurons': [1, 1000],
                      'activation': ['relu', 'elu', 'selu', 'linear'],
                      'optimizer': ['sgd', 'adam'],
                      'epsilon': [1, 1000],
                      'zeta': [1 * 10**(-1), 9 * 10**(-1)],
                      'learning_rate': [10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5)],
                      'maxepoches' : [50, 5000],
                      'SETepoches': [1, 1000],
                      'conserveMass': [False]}

select_possible_genes = {'nb_layers': 'int',
                         'nb_neurons': 'int',
                         'activation': 'list',
                         'optimizer': 'list',
                         'epsilon': 'real',
                         'zeta': 'real',
                         'learning_rate': 'list',
                         'maxepoches' : 'int',
                         'SETepoches' : 'int',
                         'conserveMass': 'list'}


GENOME_FILENAME = 'Genome_GeneticAlgorithm_{:0>3d}_Generation_{:0>2d}_ID_{:0>2d}.pickle'
RE_PATTERN_GENOME_FILENAME = r'^Genome_GeneticAlgorithm_(\d+)_Generation_(\d+)_ID_(\d+).pickle'

# Directory names
GENETIC_ALGORITHM_DIRECTORY = 'GeneticAlgorithm'
GENETIC_ALGORITHM = 'GeneticAlgorithm_{:0>3d}'
GENETIC_ALGORITHM_MODEL = 'Generation_{:0>2d}_ID_{:0>2d}'

# Pattern joboutput
PATTERN_LOGFILE_GENETIC_ALGORITHM = 'Logfile_GeneticAlgorithm_{:0>3d}.log'
PATTERN_LOGFILE_TRANING_GENOME = 'Logfile_GeneticAlgorithm_{:0>3d}_Generation_{:0>2d}_TrainGenome_{:0>2d}.txt'
PATTERN_JOBFILE_TRANING_GENOME = 'Jobfile_GeneticAlgorithm_{:0>3d}_Generation_{:0>2d}_TrainGenome_{:0>2d}.txt'
PATTERN_JOBOUTPUT_TRANING_GENOME = 'Joboutput_GeneticAlgorithm_{:0>3d}_Generation_{:0>2d}_TrainGenome_{:0>2d}.txt'

PATTERN_TARFILE = 'GeneticAlgorithm_{:0>3d}_Generation.tar.{}'
PATTERN_TARFILE_GENERATION = 'GeneticAlgorithm_{:0>3d}_Generation_{:0>3d}.tar.{}'

