#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import logging
import sqlite3
import traceback

import initialValue.constants as InitialValue_Constants
from initialValue.InitialValueSimulation import InitialValueSimulation
import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants


def main(metos3dModel, parameterId=0, concentrationTyp='constant', concentrationNum=0, differentTracer=None, timestep=1, nodes=NeshCluster_Constants.DEFAULT_NODES):
    """
    Starts spin up simulation using different intial concentration

    Parameters
    ----------
    metos3dModel : str
        Name of the biogeochemical model
    parameterId : int, default: 0
        Id of the parameter of the latin hypercube example
    concentrationTyp : {'vector', 'constant'}, default: 'constant'
        Use constant initial concentration or an initial concentration
        defined with vectors
    concentrationNum : int, default: 0
        Index of the concentration
    differentTracer : int or None, default: None
        Number of different tracer
    timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
        Time step of the spin up simulation
    nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
        Number of nodes on the high performance cluster
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(parameterId) is int and parameterId in range(InitialValue_Constants.PARAMETERID_MAX+1)
    assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
    assert type(concentrationNum) is int and 0 <= concentrationNum
    assert differentTracer is None or type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel])
    assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
    assert type(nodes) is int and 0 < nodes

    #Logging
    logfile = os.path.join(InitialValue_Constants.PATH, 'Data', 'Logs', 'Logfile', InitialValue_Constants.PATTERN_LOGFILE.format(metos3dModel, parameterId, concentrationTyp, concentrationNum, differentTracer if differentTracer is not None else len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel]), timestep))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=logfile, filemode='a', level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        initialValueSimulation = InitialValueSimulation(metos3dModel, parameterId=parameterId, timestep=timestep, concentrationTyp=concentrationTyp, concentrationNum=concentrationNum)
        if differentTracer is not None:
            initialValueSimulation.set_differentTracer(differentTracer)

        #Spin up simulation using Metos3d
        if not initialValueSimulation.existsMetos3dOutput():
            initialValueSimulation.set_nodes(nodes=nodes)
            initialValueSimulation.set_removeTracer()
            initialValueSimulation.run()

        #Insert the results of the spin up into the database
        initialValueSimulation.evaluation()

        initialValueSimulation.close_DB_connection()

    except AssertionError as err:
        logging.error('Assertion error for\nMetos3dModel: {:s}\nParameterId: {:d}\nConcentrationTyp: {:s}\nConcentrationNum: {:d}\n{}'.format(metos3dModel, parameterId, concentrationTyp, concentrationNum, err))
        traceback.print_exc()
    except sqlite3.DatabaseError as err:
        logging.error('Database error for\nMetos3dModel: {:s}\nParameterId: {:d}\nConcentrationTyp: {:s}\nConcentrationNum: {:d}\n{}'.format(metos3dModel, parameterId, concentrationTyp, concentrationNum, err))
        traceback.print_exc()
    finally:
        try:
            initialValueSimulation.close_DB_connection()
        except UnboundLocalError as ule:
            pass



if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-metos3dModel', type=str, help='Name of the biogeochemical model')
    parser.add_argument('-parameterId', nargs='?', type=int, const=0, default=0, help='Id of the parameter of the latin hypercube example')
    parser.add_argument('-concentrationTyp', nargs='?', type=str, const='constant', default='const', help='Use constant initial concentration or an initial concentration defined with vectors')
    parser.add_argument('-concentrationNum', nargs='?', type=int, const=0, default=0, help='Index of the concentration')
    parser.add_argument('-differentTracer', nargs='?', const=None, default=None, help='Number of different tracer')
    parser.add_argument('-timestep', nargs='?', type=int, const=1, default=1, help='Time step of the spin up simulation')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')

    args = parser.parse_args()
    differentTracer = None if args.differentTracer is None else int(args.differentTracer)

    main(args.metos3dModel, parameterId=args.parameterId, concentrationTyp=args.concentrationTyp, concentrationNum=args.concentrationNum, differentTracer=differentTracer, timestep=args.timestep, nodes=args.nodes)

