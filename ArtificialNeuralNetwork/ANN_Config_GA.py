#!/usr/bin/env python
# -*- coding: utf8 -*

import ann.network.constants as ANN_Constants

ANN_Config_Default = {'algorithm': 'GeneticAlgorithm',
                      'generations': 10,
                      'populationSize': 30,
                      'metos3dModel': 'N',
                      'config': {},
                      'gaParameter': {}}

#The gaParameter controls the geneticAlgorithm. Using the GeneticAlgorithm 'retain', 'random_select' and 'mutate_change' arepossible parameter. Using the genetic algorithm of Rechenberg 'alpha' and 'offspring' are possible parameter.
#The config parameters controls the training of the neural networks. Possible entries are 'indexMin', 'indexMax' and 'trainingSize'.

ANN_Config_GA = {}

ANN_Config_GA[0] = {'algorithm': 'GeneticAlgorithm',
                    'generations': 10,
                    'populationSize': 30,
                    'config': {'indexMin': 0,
                               'indexMax': 1100}}

ANN_Config_GA[1] = {'algorithm': 'Rechenberg',
                    'generations': 50,
                    'populationSize': 10,
                    'config': {'indexMin': ANN_Constants.PARAMETERID_MAX_TEST+1,
                               'indexMax': ANN_Constants.PARAMETERID_MAX}}

ANN_Config_GA[2] = {'algorithm': 'GeneticAlgorithm',
                    'generations': 10,
                    'populationSize': 30,
                    'config': {'indexMin': ANN_Constants.PARAMETERID_MAX_TEST+1,
                               'indexMax': ANN_Constants.PARAMETERID_MAX}}

ANN_Config_GA[5] = {'algorithm': 'GeneticAlgorithm',
                    'generations': 10,
                    'populationSize': 30,
                    'config': {'indexMin': ANN_Constants.PARAMETERID_MAX_TEST+1,
                               'indexMax': 1000,
                               'trainingSize': 100}}

ANN_Config_GA[6] = {'algorithm': 'Rechenberg',
                    'generations': 50,
                    'populationSize': 10,
                    'config': {'indexMin': ANN_Constants.PARAMETERID_MAX_TEST+1,
                               'indexMax': 1000,
                               'trainingSize': 100}}

ANN_Config_GA[7] = {'algorithm': 'GeneticAlgorithm',
                    'generations': 10,
                    'populationSize': 30,
                    'config': {'indexMin': ANN_Constants.PARAMETERID_MAX_TEST+1,
                               'indexMax': 1100}}

ANN_Config_GA[8] = {'algorithm': 'Rechenberg',
                    'generations': 50,
                    'populationSize': 10,
                    'config': {'indexMin': ANN_Constants.PARAMETERID_MAX_TEST+1,
                               'indexMax': 1100}}

