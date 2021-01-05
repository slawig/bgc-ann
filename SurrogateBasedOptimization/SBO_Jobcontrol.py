#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import logging

import neshCluster.constants as NeshCluster_Constants
import sbo.constants as SBO_Constants
from sbo.SurrogateBasedOptimization import SurrogateBasedOptimization
import SBO_Config


def main(optimizationId, queue=NeshCluster_Constants.DEFAULT_QUEUE, cores=NeshCluster_Constants.DEFAULT_CORES):
    """
    Start the optimization using surrogate based optimization
    @author: Markus Pfeil
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and 0 < cores

    filename = os.path.join(SBO_Constants.PATH, 'Optimization', 'Logfile', SBO_Constants.PATTERN_LOGFILE.format(optimizationId))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=filename, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    #Options parameter for the optimization
    config = parseConfig(optimizationId)

    #Start the optimization
    sbo = SurrogateBasedOptimization(optimizationId, queue=queue, cores=cores)
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
    parser.add_argument("-queue", nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_QUEUE, default=NeshCluster_Constants.DEFAULT_QUEUE, help="Queue of the nesh cluster to run the job")
    parser.add_argument("-cores", nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_CORES, default=NeshCluster_Constants.DEFAULT_CORES, help="Number of cores for the job")

    args = parser.parse_args()

    main(optimizationId=args.optimizationId, queue=args.queue, cores=args.cores)

