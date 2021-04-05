#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os

import initialValue.constants as InitialValue_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import neshCluster.constants as NeshCluster_Constants

from system.system import SYSTEM
if SYSTEM == 'PC':
    from standaloneComputer.JobAdministration import JobAdministration
else:
    from neshCluster.JobAdministration import JobAdministration


def main(metos3dModel, parameterIdList=[0], concentrationTyp='constant', concentrationNumList=range(100), differentTracer=None, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
    """
    Create jobs for the initial value simulation

    Parameters
    ----------
    metos3dModel : str
        Name of the biogeochemical model
    parameterIdList : list [int], default: [0]
        List of parameterIds of the latin hypercube sample
    concentrationTyp : {'vector', 'constant'}, default: 'constant'
        Use constant initial concentration or an initial concentration
        defined with vectors
    concentrationNumList : list [int], default: range(100)
        List of concentration indices
    differentTracer : int or None, default: None
        Number of different tracer
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
    assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
    assert type(concentrationNumList) in [list, range]
    assert differentTracer is None or type(differentTracer) is int and 1 <= differentTracer and differentTracer <= len(Metos3d_Constants.METOS3D_MODEL_TRACER[metos3dModel])
    assert partition in NeshCluster_Constants.PARTITION
    assert qos in NeshCluster_Constants.QOS
    assert type(nodes) is int and 0 < nodes
    assert memory is None or type(memory) is int and 0 < memory
    assert time is None or type(time) is int and 0 < time
    
    initialValueSimulation = InitialValueSimulation(metos3dModel, concentrationNumList=concentrationNumList, concentrationTyp=concentrationTyp, parameterIdList=parameterIdList, partition=partition, qos=qos, nodes=nodes, memory=memory, time=time)

    if differentTracer is not None:
        initialValueSimulation.set_differentTracer(differentTracer)

    initialValueSimulation.generateJobList()
    initialValueSimulation.runJobs()



class InitialValueSimulation(JobAdministration):
    """
    Administration of the jobs organizing the initial value simulations
    """

    def __init__(self, metos3dModel, concentrationNumList=range(100), concentrationTyp='constant', parameterIdList=[0], partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
        """
        Initializes the jobs of the initial value simulation

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        concentrationNumList : list [int], default: range(100)
            List of concentration indices
        concentrationTyp : {'vector', 'constant'}, default: 'constant'
            Use constant initial concentration or an initial concentration
            defined with vectors
        parameterIdList : list [int], default: [0]
            List of parameterIds of the latin hypercube sample
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
        assert type(concentrationNumList) in [list, range]
        assert concentrationTyp in Metos3d_Constants.METOS3D_MODEL_TRACER_CONCENTRATIONTYP
        assert type(parameterIdList) in [list, range]
        assert partition in NeshCluster_Constants.PARTITION
        assert qos in NeshCluster_Constants.QOS
        assert type(nodes) is int and 0 < nodes
        assert memory is None or type(memory) is int and 0 < memory
        assert time is None or type(time) is int and 0 < time

        JobAdministration.__init__(self)

        self._metos3dModel = metos3dModel
        self._timestep = 1
        self._concentrationNumList = concentrationNumList
        self._differentTracer = len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])
        self._concentrationTyp = concentrationTyp
        self._parameterIdList = parameterIdList
        self._partition = partition
        self._qos = qos
        self._nodes = nodes
        self._memory = memory
        self._time = time


    def set_differentTracer(self, differentTracer):
        """
        Set the number of different tracer

        Parameters
        ----------
        differentTracer : int
            Number of different tracer
        """
        assert type(differentTracer) is int and 0 < differentTracer and differentTracer <= len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])

        self._differentTracer = differentTracer


    def generateJobList(self):
        """
        Generates a list of jobs of initial value simulations
        """
        for parameterId in self._parameterIdList:
            for concentrationNum in self._concentrationNumList:
                if not self._checkJob(parameterId, concentrationNum):
                    program = 'InitialValue_Jobcontrol.py -metos3dModel {:s} -parameterId {:d} -concentrationTyp {:s} -concentrationNum {:d} -differentTracer {:d} -timestep {:d}'.format(self._metos3dModel, parameterId, self._concentrationTyp, concentrationNum, self._differentTracer, self._timestep)

                    jobDict = {}
                    jobDict['jobFilename'] = os.path.join(InitialValue_Constants.PATH, 'Jobfile', InitialValue_Constants.PATTERN_JOBFILE.format(self._metos3dModel, parameterId, self._concentrationTyp, concentrationNum, self._differentTracer, self._timestep))
                    jobDict['path'] = os.path.join(InitialValue_Constants.PATH, 'Jobfile')
                    jobDict['jobname'] = 'InitialValue_{:d}_{}'.format(concentrationNum, self._metos3dModel)
                    jobDict['joboutput'] = os.path.join(InitialValue_Constants.PATH, 'Data', 'Logs', 'Joboutput', InitialValue_Constants.PATTERN_JOBOUTPUT.format(self._metos3dModel, parameterId, self._concentrationTyp, concentrationNum, self._differentTracer, self._timestep))
                    jobDict['programm'] = os.path.join(InitialValue_Constants.PROGRAM_PATH, program)
                    jobDict['partition'] = self._partition
                    jobDict['qos'] = self._qos
                    jobDict['nodes'] = self._nodes
                    jobDict['pythonpath'] = InitialValue_Constants.DEFAULT_PYTHONPATH
                    jobDict['loadingModulesScript'] = NeshCluster_Constants.DEFAULT_LOADING_MODULES_SCRIPT

                    if self._memory is not None:
                        jobDict['memory'] = self._memory
                    if self._time is not None:
                        jobDict['time'] = self._time

                    self.addJob(jobDict)


    def _checkJob(self, parameterId, concentrationNum):
        """
        Check if the job run already exists

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        concentrationNum : int
            Index of the concentration

        Returns
        -------
        bool
           True if the joboutput already exists
        """
        assert type(parameterId) is int and parameterId in range(InitialValue_Constants.PARAMETERID_MAX+1)
        assert type(concentrationNum) is int and 0 <= concentrationNum

        joboutput = os.path.join(InitialValue_Constants.PATH, 'Data', 'Logs', 'Joboutput', InitialValue_Constants.PATTERN_JOBOUTPUT.format(self._metos3dModel, parameterId, self._concentrationTyp, concentrationNum, self._differentTracer, self._timestep))
        return os.path.exists(joboutput) and os.path.isfile(joboutput)



if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-metos3dModel', type=str, help='Name of the biogeochemical model')
    parser.add_argument('-concentrationTyp', nargs='?', type=str, const='constant', default='constant', help='Use constant initial concentration or an initial concentration defined with vectors')
    parser.add_argument('-concentrationNums', nargs='*', type=int, default=[], help='List of concentrationNums')
    parser.add_argument('-concentrationNumRange', nargs=2, type=int, default=[], help='Create list of concentationNums using range (-concentrationNumRange a b: range(a, b))')
    parser.add_argument('-parameterIds', nargs='*', type=int, default=[0], help='List of parameterIds')
    parser.add_argument('-parameterIdRange', nargs=2, type=int, default=[0], help='Create list parameterIds using range (-parameterIdRange a b: range(a, b)')
    parser.add_argument('-differentTracer', nargs='?', const=None, default=None, help='Number of different tracer')
    parser.add_argument('-partition', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_PARTITION, default=NeshCluster_Constants.DEFAULT_PARTITION, help='Partition of slum on the Nesh-Cluster (Batch class)')
    parser.add_argument('-qos', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_QOS, default=NeshCluster_Constants.DEFAULT_QOS, help='Quality of service on the Nesh-Cluster')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')
    parser.add_argument('-memory', nargs='?', type=int, const=None, default=None, help='Memory in GB for the job on the Nesh-Cluster')
    parser.add_argument('-time', nargs='?', type=int, const=None, default=None, help='Time in hours for the job on the Nesh-Cluster')

    args = parser.parse_args()
    concentrationNumList = args.concentrationNums if len(args.concentrationNums) > 0 or len(args.concentrationNumRange) != 2 else range(args.concentrationNumRange[0], args.concentrationNumRange[1])
    parameterIdList = args.parameterIds if len(args.parameterIds) > 0 or len(args.parameterIdRange) != 2 else range(args.parameterIdRange[0], args.parameterIdRange[1])
    differentTracer = None if args.differentTracer is None else int(args.differentTracer)

    main(args.metos3dModel, parameterIdList=parameterIdList, concentrationTyp=args.concentrationTyp, concentrationNumList=concentrationNumList, differentTracer=differentTracer, partition=args.partition, qos=args.qos, nodes=args.nodes, memory=args.memory, time=args.time)

