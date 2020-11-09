#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import logging
import os
import sqlite3
import traceback

import ann.network.constants as ANN_Constants
import neshCluster.constants as NeshCluster_Constants
import ann.evaluation.constants as Evaluation_Constants
from ann.evaluation.PredictionEvaluation import PredictionEvaluation
from ann.evaluation.DatabaseInsertEvaluation import DatabaseInsertEvaluation
from ann.database.access import Ann_Database


def main(annId, parameterId=0, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2, remove=True, trajectoryFlag=True):
    """
    Start the simultion with metos3d using the calculated tracer concentration of the artificial neural network with the given annId as initial concentration
    @author: Markus Pfeil
    """
    assert annId in range(0, ANN_Constants.ANNID_MAX+1) 
    assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or (type(tolerance) is float and tolerance > 0)
    assert type(spinupToleranceReference) is bool and ((not spinupToleranceReference) or (spinupToleranceReference and tolerance > 0))
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and cores > 0
    assert type(remove) is bool
    assert type(trajectoryFlag) is bool
    
    #Configure of the logging
    annDb = Ann_Database()
    annType, model = annDb.get_annTypeModel(annId)
    annDb.close_connection()
    if spinupToleranceReference:
        filename = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Logfile', Evaluation_Constants.PATTERN_LOGFILE_SPINUP_REFERENCE.format(model, parameterId, 0.0 if tolerance is None else tolerance, cores * NeshCluster_Constants.CPUNUM[queue]))
    else:
        filename = os.path.join(ANN_Constants.PATH, 'Prediction', 'Logfile', Evaluation_Constants.PATTERN_LOGFILE.format(annType, model, annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, cores * NeshCluster_Constants.CPUNUM[queue]))
    logging.basicConfig(filename=filename, filemode='w', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    years = 10000 if tolerance is not None else 1000
    trajectoryYear = 50 if tolerance is not None else 10

    try:   
        #Save the prediction and calculate the approximations as a spin up using the prediction/mass-corrected prediction as initial concentration 
        predictionEvaluation = PredictionEvaluation(annId, parameterId=parameterId, years=years, trajectoryYear=trajectoryYear, massAdjustment=massAdjustment, tolerance=tolerance, spinupToleranceReference=spinupToleranceReference, queue=queue, cores=cores)
        predictionEvaluation.run(remove=remove)
        predictionEvaluation.close_DB_connection()

        #Insert the results of the approximations into the database
        dbinsert = DatabaseInsertEvaluation(annId, parameterId=parameterId, years=years, trajectoryYear=trajectoryYear, massAdjustment=massAdjustment, tolerance=tolerance, spinupToleranceReference=spinupToleranceReference, cpunum=cores*NeshCluster_Constants.CPUNUM[queue], queue=queue, cores=cores)

        #Insert spin up norm values
        if dbinsert.existsJoboutputFile() and not dbinsert.checkSpinupTotalityDatabase():
            dbinsert.insertSpinup(overwrite=True)

        #Insert mass values
        if dbinsert.existsJoboutputFile() and not dbinsert.checkMassTotalityDatabase():
            dbinsert.calculateMass(overwrite=True)

        #Insert norm values
        if dbinsert.existsJoboutputFile() and (not dbinsert.checkNormTotalityDatabase() or not dbinsert.checkDeviationTotalityDatabase()):
            dbinsert.calculateNorm(overwrite=True)

        #Insert trajectory norm values
        if trajectoryFlag and dbinsert.existsJoboutputFile() and not dbinsert.checkTrajectoryNormTotalityDatabase():
            dbinsert.calculateTrajectoryNorm(overwrite=True)

        #Insert ann training progress
        if dbinsert.checkAnnTraining():
            dbinsert.insertTraining(overwrite=True)

        dbinsert.close_DB_connection()

    except AssertionError as err:
        logging.error('Assertion error for annId {:0>5d} and parameterId {:0>3d}\n{}'.format(annId, parameterId, err))
        traceback.print_exc()
    except sqlite3.DatabaseError as err:
        logging.error('Database error for annId {:0>5d} and parameterId {:0>3d}\n{}'.format(annId, parameterId, err))
        traceback.print_exc()
    finally:
        try:
            predictionEvaluation.close_DB_connection()
            dbinsert.close_DB_connection()
        except UnboundLocalError as ule:
            logging.error('No DbInsert instance: {}', ule)


if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("annId", type=int, help="Id of the artificial neural network (see database table AnnConfig)")
    parser.add_argument("parameterId", type=int, help="Parameter id for the model parameter (see database table Parameter)")
    parser.add_argument("-tolerance", nargs='?', const=None, default=None, help="Tolerance for the spin up calculation")
    parser.add_argument("--massAdjustment", "--massAdjustment", action="store_true")
    parser.add_argument("--spinupToleranceReference", "--spinupToleranceReference", action="store_true")
    parser.add_argument("-queue", nargs='?', type=str, const='clmedium', default='clmedium', help="Queue of the nesh cluster to run the job")
    parser.add_argument("-cores", nargs='?', type=int, const=2, default=2, help="Number of cores for the job")
    parser.add_argument("--trajectory", "--trajectory", action="store_true")

    args = parser.parse_args()
    tolerance = None if args.tolerance is None else float(args.tolerance)

    main(annId=args.annId, parameterId=args.parameterId, massAdjustment=args.massAdjustment, tolerance=tolerance, spinupToleranceReference=args.spinupToleranceReference, queue=args.queue, cores=args.cores, trajectoryFlag=args.trajectory)

