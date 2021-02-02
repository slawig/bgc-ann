#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import logging

import neshCluster.constants as NeshCluster_Constants
import sbo.constants as SBO_Constants
from sbo.SurrogateBasedOptimization import SurrogateBasedOptimization
import SBO_Config


def main(optimizationId, nodes=NeshCluster_Constants.DEFAULT_NODES):
    """
    Start the optimization using surrogate based optimization
    @author: Markus Pfeil
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert type(nodes) is int and 0 < nodes

    filename = os.path.join(SBO_Constants.PATH, 'Optimization', 'Logfile', SBO_Constants.PATTERN_LOGFILE.format(optimizationId))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=filename, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    #Options parameter for the optimization
    config = parseConfig(optimizationId)

    #Start the optimization
    sbo = SurrogateBasedOptimization(optimizationId, nodes=nodes)
    u = sbo.run(options=config['Options'])

    logging.info('****Optimal parameter: {}****'.format(u))
    
    #Create backup
    logging.debug('***Create backup of the simulation data***')
    sbo.backup()
    sbo.close_connection()


def parseConfig(optimizationId):
    """
    Parse the config for the given optimizationId
    @author: Markus Pfeil
    """
    assert type(optimizationId) is int and 0 <= optimizationId    

    config = {}
    for key in SBO_Config.SBO_Config_Default:
        try:
            config[key] = SBO_Config.SBO_Config[optimizationId][key]
        except KeyError:
            config[key] = SBO_Config.SBO_Config_Default[key]

    return config




if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizationId", type=int, help="Id of the optimization")
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')

    args = parser.parse_args()

    main(optimizationId=args.optimizationId, nodes=args.nodes)

