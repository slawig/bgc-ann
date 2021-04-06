#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import sqlite3

import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsDatabase import Timesteps_Database


from system.system import SYSTEM
if SYSTEM == 'PC':
    from standaloneComputer.JobAdministration import JobAdministration
else:
    from neshCluster.JobAdministration import JobAdministration


def main(metos3dModel, parameterIdList=range(Timesteps_Constants.PARAMETERID_MAX+1), timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, concentrationId=None, convergence=False, oscillation=False, spinupTolerance=None, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
    """
    Create jobs for the spin ups using different time steps

    Parameters
    ----------
    metos3dModel : str
        Name of the biogeochemical model
    parameterIdList : list [int] or range,
        default: range(Timesteps_Constants.PARAMETERID_MAX+1)
        List of parameterIds of the latin hypercube example
    timestepList : list, default: [1, 2, 4, 8, 16, 32, 64]
        List of time steps of the spin up simulation
    concentrationId : int or None, default: None
        Id of the initial tracer concentration
    convergence : bool, default: False
        If True, check the simulation using the standard constant
        concenration for convergence of the spin up
    oscillation : bool, default: False
        If True, generates job file with another initial concentration
        where the simulation oscillates with the standard constant
        concentration
    spinupTolerance : float or None, default: None
        If not None, check if the spin up tolerance of the simulation
        using the standard constant concentration is less than the given
        spinupTolerance value
    partition : str, default: NeshCluster_Constants.DEFAULT_PARTITION
        Partition of the NEC HPC-Linux-Cluster of the CAU Kiel
    qos : str, default: NeshCluster_Constants.DEFAULT_QOS
        Quality of service of the NEC HPC-Linux-Cluster of the CAU Kiel
    nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
        Number of nodes on the NEC HPC-Linux-Cluster of the CAU Kiel
    memory : int or None, default: None
        Reserved memory on the NEC HPC-Linux-Cluster of the CAU Kiel
    time : int or None, default: None
        Walltime in hours on the NEC HPC-Linux-Cluster of the CAU Kiel
    """
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    assert type(parameterIdList) in [list, range]
    assert type(timestepList) in [list, range]
    assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId
    assert type(convergence) is bool
    assert type(oscillation) is bool
    assert spinupTolerance is None or type(spinupTolerance) is float and 0 < spinupTolerance
    assert partition in NeshCluster_Constants.PARTITION
    assert qos in NeshCluster_Constants.QOS
    assert type(nodes) is int and 0 < nodes
    assert memory is None or type(memory) is int and 0 < memory
    assert time is None or type(time) is int and 0 < time

    timestepsSimulation = TimestepsSimulation(metos3dModel, parameterIdList=parameterIdList, timestepList=timestepList, concentrationId=concentrationId, partition=partition, qos=qos, nodes=nodes, memory=memory, time=time)
    timestepsSimulation.generateJobList(convergence=convergence, oscillation=oscillation, spinupTolerance=spinupTolerance)
    timestepsSimulation.runJobs()



class TimestepsSimulation(JobAdministration):
    """
    Administration of the jobs organizing spin ups using different time steps
    """

    def __init__(self, metos3dModel, parameterIdList=range(Timesteps_Constants.PARAMETERID_MAX+1), timestepList=Metos3d_Constants.METOS3D_TIMESTEPS, concentrationId=None, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
        """
        Initializes the jobs of the spin ups using different time steps

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterIdList : list [int] or range,
            default: range(Timesteps_Constants.PARAMETERID_MAX+1)
            List of parameterIds of the latin hypercube example
        timestepList : list, default: [1, 2, 4, 8, 16, 32, 64]
            List of time steps of the spin up simulation
        concentrationId : int or None, default: None
            Id of the initial tracer concentration
        partition : str, default: NeshCluster_Constants.DEFAULT_PARTITION
            Partition of the NEC HPC-Linux-Cluster of the CAU Kiel
        qos : str, default: NeshCluster_Constants.DEFAULT_QOS
            Quality of service of the NEC HPC-Linux-Cluster of the CAU Kiel
        nodes : int, default: NeshCluster_Constants.DEFAULT_NODES
            Number of nodes on the NEC HPC-Linux-Cluster of the CAU Kiel
        memory : int or None, default: None
            Reserved memory on the NEC HPC-Linux-Cluster of the CAU Kiel
        time : int or None, default: None
            Walltime in hours on the NEC HPC-Linux-Cluster of the CAU Kiel
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterIdList) in [list, range]
        assert type(timestepList) in [list, range]
        assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId
        assert partition in NeshCluster_Constants.PARTITION
        assert qos in NeshCluster_Constants.QOS
        assert type(nodes) is int and 0 < nodes
        assert memory is None or type(memory) is int and 0 < memory
        assert time is None or type(time) is int and 0 < time

        JobAdministration.__init__(self)

        self._metos3dModel = metos3dModel
        self._parameterIdList = parameterIdList
        self._timestepList = timestepList
        self._concentrationId = concentrationId

        self._partition = partition
        self._qos = qos
        self._nodes = nodes
        self._memory = memory
        self._time = time


    def generateJobList(self, convergence=False, oscillation=False, spinupTolerance=None):
        """
        Generates a list of jobs of spin ups using different time steps

        Parameters
        ----------
        convergence : bool, default: False
            If True, check the simulation using the standard constant
            concenration for convergence of the spin up
        oscillation : bool, default: False
            If True, generates job file with another initial concentration
            where the simulation oscillates with the standard constant
            concentration
        spinupTolerance : float or None, default: None
            If not None, check if the spin up tolerance of the simulation
            using the standard constant concentration is less than the given
            spinupTolerance value

        Notes
        -----
        Adds the jobs into the internal job list
        """
        assert type(convergence) is bool
        assert type(oscillation) is bool
        assert spinupTolerance is None or type(spinupTolerance) is float and 0 < spinupTolerance

        for parameterId in self._parameterIdList:
            for timestep in self._timestepList:
                if not self._checkJob(parameterId, timestep) and self._checkConvergenceOscillation(parameterId, timestep, convergence=convergence, oscillation=oscillation, spinupTolerance=spinupTolerance):
                    program = 'Timesteps_Jobcontrol.py -metos3dModel {:s} -parameterId {:d} -timestep {:d} -nodes {:d}'.format(self._metos3dModel, parameterId, timestep, self._nodes)

                    #Optional parameter
                    if self._concentrationId is not None:
                        program += ' -concentrationId {:d}'.format(self._concentrationId)

                    jobDict = {}
                    jobDict['jobFilename'] = os.path.join(Timesteps_Constants.PATH, 'Timesteps', 'Jobfile', Timesteps_Constants.PATTERN_JOBFILE.format(self._metos3dModel, parameterId, timestep, self._concentrationId if self._concentrationId is not None else Timesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel]))
                    jobDict['path'] = os.path.join(Timesteps_Constants.PATH, 'Timesteps', 'Jobfile')
                    jobDict['jobname'] = 'Timesteps_{}_{:d}_{:d}'.format(self._metos3dModel, parameterId, timestep)
                    jobDict['joboutput'] = os.path.join(Timesteps_Constants.PATH, 'Timesteps', 'Joboutput', Timesteps_Constants.PATTERN_JOBOUTPUT.format(self._metos3dModel, parameterId, timestep, self._concentrationId if self._concentrationId is not None else Timesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel]))
                    jobDict['programm'] = os.path.join(Timesteps_Constants.PROGRAM_PATH, program)
                    jobDict['partition'] = self._partition
                    jobDict['qos'] = self._qos
                    jobDict['nodes'] = self._nodes
                    jobDict['pythonpath'] = Timesteps_Constants.DEFAULT_PYTHONPATH
                    jobDict['loadingModulesScript'] = NeshCluster_Constants.DEFAULT_LOADING_MODULES_SCRIPT

                    if self._memory is not None:
                        jobDict['memory'] = self._memory
                    if self._time is not None:
                        jobDict['time'] = self._time

                    self.addJob(jobDict)


    def _checkJob(self, parameterId, timestep):
        """
        Check if the job run already exists

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}
            Time step of the spin up simulation

        Returns
        -------
        bool
           True if the joboutput already exists
        """
        assert type(parameterId) is int and parameterId in range(Timesteps_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        joboutput = os.path.join(Timesteps_Constants.PATH, 'Timesteps', 'Joboutput', Timesteps_Constants.PATTERN_JOBOUTPUT.format(self._metos3dModel, parameterId, timestep, self._concentrationId if self._concentrationId is not None else Timesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel]))
        return os.path.exists(joboutput) and os.path.isfile(joboutput)


    def _checkConvergenceOscillation(self, parameterId, timestep, convergence=False, oscillation=False, spinupTolerance=None):
        """
        Check if the simulation oscillates, does not converge or does not
        reach the given spin up tolerance using the standard constant
        concenetration

        Parameters
        ----------
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation
        convergence : bool, default: False
            If True, check the simulation using the standard constant
            concenration for convergence of the spin up
        oscillation : bool, default: False
            If True, check the simulation using the standard constant
            concenration for oscillation
        spinupTolerance : float or None, default: None
            If not None, check if the spin up tolerance of the simulation
            using the standard constant concentration is less than the given
            spinupTolerance value

        Returns
        -------
        bool
            True, if the simulation does not converge, oscillates or does not
            reach the given spin up tolerance using the standard constant
            concentration

        Notes
        -----
        Inserts an entry in the database table Simulation for the simulation
        using the given concentrationId instead of the standard constant
        concentration.
        """
        assert type(parameterId) is int and parameterId in range(Timesteps_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(convergence) is bool
        assert type(oscillation) is bool
        assert spinupTolerance is None or type(spinupTolerance) is float and 0 < spinupTolerance

        divergentSimulation = not convergence
        oscillatingSimulation = not oscillation
        spinupToleranceSimulation = True if spinupTolerance is None else False

        if convergence or oscillation or spinupTolerance is not None:
            divergentSimulation = False
            oscillatingSimulation = False
            spinupToleranceSimulation = False

            database = Timesteps_Database()

            try:
                simulationId = database.get_simulationId(self._metos3dModel, parameterId, Timesteps_Constants.CONCENTRATIONID_DICT[self._metos3dModel], timestep=timestep)

                #Diverges the simulation
                if convergence:
                    divergentSimulation = not database.get_convergence(simulationId)

                #Oscillates the simulation
                if oscillation:
                    oscillatingSimulation = database.oscillation_spin(simulationId)

                #Reaches the simulation the spin up tolerance
                if spinupTolerance is not None:
                    spinupToleranceSimulation = (database.read_spinupTolerance(simulationId) > spinupTolerance)

                #Insert simulation in the database
                if (divergentSimulation or oscillatingSimulation or spinupToleranceSimulation) and not database.exists_simulaiton(self._metos3dModel, parameterId, self._concentrationId, timestep=timestep):
                    database.insert_simulation(self._metos3dModel, parameterId, self._concentrationId, timestep=timestep)

            except (AssertionError, sqlite3.OperationalError) as err:
                divergentSimulation = False
                oscillatingSimulation = False
                spinupToleranceSimulation = False

            database.close_connection()

        return divergentSimulation or oscillatingSimulation or spinupToleranceSimulation



if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-metos3dModel', type=str, help='Name of the biogeochemical model')
    parser.add_argument('-parameterIds', nargs='*', type=int, default=[0], help='List of parameterIds')
    parser.add_argument('-parameterIdRange', nargs=2, type=int, default=[0], help='Create list parameterIds using range (-parameterIdRange a b: range(a, b)')
    parser.add_argument('-timesteps', nargs='*', type=int, default=Metos3d_Constants.METOS3D_TIMESTEPS, help='List of time steps for the spin up simulation')
    parser.add_argument('-concentrationId', nargs='?', const=None, default=None, help='Id of the initial concentration')
    parser.add_argument('-convergence', '--convergence', action='store_true', help='Simulation with another initial concentration for divergent simulations')
    parser.add_argument('-oscillation', '--oscillation', action='store_true', help='Simulation with another initial concentration for oscillating simulations')
    parser.add_argument('-spinupTolerance', nargs='?', const=None, default=None, help='Simulation with another initial concentration if the simulation using standard concentration does not reach the spin up tolerance')
    parser.add_argument('-partition', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_PARTITION, default=NeshCluster_Constants.DEFAULT_PARTITION, help='Partition of slum on the Nesh-Cluster (Batch class)')
    parser.add_argument('-qos', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_QOS, default=NeshCluster_Constants.DEFAULT_QOS, help='Quality of service on the Nesh-Cluster')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')
    parser.add_argument('-memory', nargs='?', type=int, const=None, default=None, help='Memory in GB for the job on the Nesh-Cluster')
    parser.add_argument('-time', nargs='?', type=int, const=None, default=None, help='Time in hours for the job on the Nesh-Cluster')

    args = parser.parse_args()
    parameterIdList = args.parameterIds if len(args.parameterIdRange) != 2 else range(args.parameterIdRange[0], args.parameterIdRange[1])

    concentrationId = None if args.concentrationId is None else int(args.concentrationId)
    spinupTolerance = None if args.spinupTolerance is None else float(args.spinupTolerance)

    main(args.metos3dModel, parameterIdList=parameterIdList, timestepList=args.timesteps, concentrationId=concentrationId, convergence=args.convergence, oscillation=args.oscillation, spinupTolerance=spinupTolerance, partition=args.partition, qos=args.qos, nodes=args.nodes, memory=args.memory, time=args.time)

