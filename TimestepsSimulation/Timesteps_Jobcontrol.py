#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import logging
import sqlite3
import traceback

import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsSimulation import TimestepsSimulation


def main(metos3dModel, parameterId=0, timestep=1, concentrationId=None, nodes=NeshCluster_Constants.DEFAULT_NODES):
    """
    Starts spin up simulation using different intial concentration

    Parameters
    ----------
    metos3dModel : str
        Name of the biogeochemical model
    parameterId : int, default: 0
        Id of the parameter of the latin hypercube example
    timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
        Time step of the spin up simulation
    nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
        Number of nodes on the high performance cluster
    concentrationId : int or None, default: None
        Id of the initial tracer concentration
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(parameterId) is int and parameterId in range(Timesteps_Constants.PARAMETERID_MAX+1)
    assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
    assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId
    assert type(nodes) is int and 0 < nodes

    #Logging
    logfile = os.path.join(Timesteps_Constants.PATH, 'Timesteps', 'Logfile', Timesteps_Constants.PATTERN_LOGFILE.format(metos3dModel, parameterId, timestep, concentrationId if concentrationId is not None else Timesteps_Constants.CONCENTRATIONID_DICT[metos3dModel]))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename=logfile, filemode='a', level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        timestepsSimulation = TimestepsSimulation(metos3dModel, parameterId=parameterId, timestep=timestep)
        if concentrationId is not None:
            timestepsSimulation.set_concentrationId(concentrationId=concentrationId)

        #Spin up simulation using Metos3d
        if not timestepsSimulation.existsMetos3dOutput():
            timestepsSimulation.set_nodes(nodes=nodes)
            timestepsSimulation.set_removeTracer()
            timestepsSimulation.run()

        #Insert the results of the spin up into the database
        timestepsSimulation.evaluation()

        timestepsSimulation.close_DB_connection()

    except AssertionError as err:
        logging.error('Assertion error for\nMetos3dModel: {:s}\nParameterId: {:d}\nTime step: {:d}\n{}'.format(metos3dModel, parameterId, timestep, err))
        traceback.print_exc()
    except sqlite3.DatabaseError as err:
        logging.error('Database error for\nMetos3dModel: {:s}\nParameterId: {:d}\nTime step: {:d}\n{}'.format(metos3dModel, parameterId, timestep, err))
        traceback.print_exc()
    finally:
        try:
            timestepsSimulation.close_DB_connection()
        except UnboundLocalError as ule:
            pass



if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-metos3dModel', type=str, help='Name of the biogeochemical model')
    parser.add_argument('-parameterId', nargs='?', type=int, const=0, default=0, help='Id of the parameter of the latin hypercube example')
    parser.add_argument('-timestep', nargs='?', type=int, const=1, default=1, help='Time step of the spin up simulation')
    parser.add_argument('-concentrationId', nargs='?', const=None, default=None, help='Id of the initial concentration')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')

    args = parser.parse_args()
    concentrationId = None if args.concentrationId is None else int(args.concentrationId)

    main(args.metos3dModel, parameterId=args.parameterId, timestep=args.timestep, concentrationId=concentrationId, nodes=args.nodes)

