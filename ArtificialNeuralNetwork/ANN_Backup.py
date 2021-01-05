#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import logging

import ann.evaluation.constants as Evaluation_Constants
from ann.evaluation.SimulationDataBackupEvaluation import SimulationDataBackupEvaluation
import ann.network.constants as ANN_Constants


def main(annId, backup=True, remove=False, restore=False, movetar=False):
    """
    Generate the backup of the simulation data generated with the artificial neural network.
    @author: Markus Pfeil
    """
    assert annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type(backup) is bool
    assert type(remove) is bool
    assert type(restore) is bool
    assert (restore and not backup and not remove) or (not restore and (backup or remove))
    assert type(movetar) is bool

    filename = Evaluation_Constants.PATTERN_BACKUP_LOGFILE.format(annId, backup, remove, restore)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=filename, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logging.info('Create backup for annId {:0>4d}'.format(annId))
    ann = SimulationDataBackupEvaluation(annId)

    #Create backup
    if backup:
        logging.info('Create backup of the simulation data generated with the artificial neural network')
        ann.backup(movetar=movetar)

    if remove:
        logging.info('Remove simulation data generated with the artificial neural network')
        ann.remove(movetar=movetar)

    #Restore the simulation using the backup
    if restore:
        logging.info('Restore the simulation data generated with the artificial neural network')
        ann.restore(movetar=movetar)

    ann.close_DB_connection()


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("annId", type=int, help="Id of the ann")
    parser.add_argument("--backup", "--backup", action="store_true", help="Create backup")
    parser.add_argument("--remove", "--remove", action="store_true", help="Remove simulation data")
    parser.add_argument("--restore", "--restore", action="store_true", help="Restore backup")
    parser.add_argument("--movetar", "--movetar", action="store_true", help="Move/Copy tarfile to/from TAPE archiv")

    args = parser.parse_args()
    main(annId=args.annId, backup=args.backup, remove=args.remove, restore=args.restore, movetar=args.movetar)

