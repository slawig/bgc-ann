#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import logging

import sbo.constants as SBO_Constants
from sbo.SurrogateBasedOptimization import SurrogateBasedOptimization

def main(optimizationId, backup=True, remove=False, restore=False, movetar=False):
    """
    Generate the backup of the SBO run
    @author: Markus Pfeil
    """
    assert type(optimizationId) is int and 0 <= optimizationId
    assert type(backup) is bool
    assert type(remove) is bool
    assert type(restore) is bool
    assert (restore and not backup and not remove) or (not restore and (backup or remove)) 
    assert type(movetar) is bool

    filename = SBO_Constants.PATTERN_BACKUP_LOGFILE.format(optimizationId, backup, remove, restore)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=filename, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    print('Create backup for optimizationId {:0>4d}'.format(optimizationId))
    sbo = SurrogateBasedOptimization(optimizationId)

    #Create backup
    if backup:
        logging.info('Create backup of the simulation data of the SBO run')
        sbo.backup(movetar=movetar)

    if remove:
        logging.info('Remove simulation data of the SBO run')
        sbo.remove(movetar=movetar)

    #Restore the simulation using the backup
    if restore:
        logging.info('Restore the simulation data of the SBO run')
        sbo.restore(movetar=movetar)

    sbo.close_connection()


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizationId", type=int, help="Id of the optimization")
    parser.add_argument("--backup", "--backup", action="store_true", help="Create backup")
    parser.add_argument("--remove", "--remove", action="store_true", help="Remove simulation data")
    parser.add_argument("--restore", "--restore", action="store_true", help="Restore backup")
    parser.add_argument("--movetar", "--movetar", action="store_true", help="Move/Copy tarfile to/from TAPE archiv")

    args = parser.parse_args()
    main(optimizationId=args.optimizationId, backup=args.backup, remove=args.remove, restore=args.restore, movetar=args.movetar)

