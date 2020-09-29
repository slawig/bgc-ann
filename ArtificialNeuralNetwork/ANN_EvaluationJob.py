#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import logging

import ann.network.constants as ANN_Constants
import neshCluster.constants as NeshCluster_Constants
import ann.evaluation.constants as Evaluation_Constants
from ann.evaluation.PredictionEvaluation import PredictionEvaluation
from ann.database.access import Ann_Database


def main(annId, parameterId=0, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2, remove=True):
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
    
    predictionEvaluation = PredictionEvaluation(annId, parameterId=parameterId, years=years, trajectoryYear=trajectoryYear, massAdjustment=massAdjustment, tolerance=tolerance, spinupToleranceReference=spinupToleranceReference, queue=queue, cores=cores)
    predictionEvaluation.run(remove=remove)


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

    args = parser.parse_args()
    tolerance = None if args.tolerance is None else float(args.tolerance)

    main(annId=args.annId, parameterId=args.parameterId, massAdjustment=args.massAdjustment, tolerance=tolerance, spinupToleranceReference=args.spinupToleranceReference, queue=args.queue, cores=args.cores)
