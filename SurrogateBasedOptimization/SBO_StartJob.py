#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os

import neshCluster.constants as NeshCluster_Constants
from database.SBO_Database import SBO_Database
import sbo.constants as SBO_Constants
from system.system import SYSTEM, PYTHON_PATH
if SYSTEM == 'PC':
    from standaloneComputer.JobAdministration import JobAdministration
else:
    from neshCluster.JobAdministration import JobAdministration


def main(optimizationIdList, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES):
    """
    Run the surrogate based optimization for the given optimizationIds.
    @author: Markus Pfeil
    """
    assert type(optimizationIdList) in [list, range]
    assert partition in NeshCluster_Constants.PARTITION
    assert qos in NeshCluster_Constants.QOS
    assert type(nodes) is int and 0 < nodes

    sbo = SurrogateBasedOptimizationJobAdministration(optimizationIdList=optimizationIdList, partition=partition, qos=qos, nodes=nodes)
    sbo.generateJobList()
    sbo.runJobs()



class SurrogateBasedOptimizationJobAdministration(JobAdministration):
    """
    Class for the administration of the jobs organizing the run of the surrogate based optimizations.
    @author: Markus Pfeil
    """

    def __init__(self, optimizationIdList, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES):
        """
        Initialisation of the evaluation jobs of the ANN with the given annId.
        @author: Markus Pfeil
        """
        assert type(optimizationIdList) in [list, range]
        assert partition in NeshCluster_Constants.PARTITION
        assert qos in NeshCluster_Constants.QOS
        assert type(nodes) is int and 0 < nodes

        JobAdministration.__init__(self)

        self._optimizationIdList = optimizationIdList
        self._partition = partition
        self._qos = qos
        self._nodes = nodes


    def generateJobList(self):
        """
        Create a list of jobs to run the surrogate based optimization.
        @author: Markus Pfeil
        """
        self._sboDb = SBO_Database()

        for optimizationId in self._optimizationIdList:
            if not self._checkJob(optimizationId):
                programm = 'SBO_Jobcontrol.py {:d} -nodes {:d}'.format(optimizationId, self._nodes)

                jobDict = {}
                jobDict['jobFilename'] = os.path.join(SBO_Constants.PATH, 'Optimization', 'Jobfile', SBO_Constants.PATTERN_JOBFILE.format(optimizationId))
                jobDict['path'] = os.path.join(SBO_Constants.PATH, 'Optimization', 'Jobfile')
                jobDict['jobname'] = 'SBO_{:d}'.format(optimizationId)
                jobDict['joboutput'] = os.path.join(SBO_Constants.PATH, 'Optimization', 'Logfile', SBO_Constants.PATTERN_JOBOUTPUT.format(optimizationId))
                jobDict['programm'] = os.path.join(SBO_Constants.PROGRAMM_PATH, programm)
                jobDict['partition'] = self._partition
                jobDict['qos'] = self._qos
                jobDict['nodes'] = self._nodes
                jobDict['memory'] = 48
                jobDict['pythonpath'] = NeshCluster_Constants.DEFAULT_PYTHONPATH
                jobDict['loadingModulesScript'] = NeshCluster_Constants.DEFAULT_LOADING_MODULES_SCRIPT

                self.addJob(jobDict)

        self._sboDb.close_connection()


    def _checkJob(self, optimizationId):
        """
        Check, if the output exists for the optimization with the given optimizationId.
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId

        existsOptimizationId = self._sboDb.exists_optimization(optimizationId)
        existsIteration = self._sboDb.exists_iteration(optimizationId)
        sboPath = os.path.join(SBO_Constants.PATH, 'Optimization', SBO_Constants.PATH_OPTIMIZATION.format(optimizationId))

        return existsOptimizationId and existsIteration and os.path.exists(sboPath) and os.path.isdir(sboPath)




if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-partition', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_PARTITION, default=NeshCluster_Constants.DEFAULT_PARTITION, help='Partition of slum on the Nesh-Cluster (Batch class)')
    parser.add_argument('-qos', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_QOS, default=NeshCluster_Constants.DEFAULT_QOS, help='Quality of service on the Nesh-Cluster')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')
    parser.add_argument('-optimizationIds', nargs='*', type=int, default=[], help='List of optimizationIds')
    parser.add_argument('-optimizationIdRange', nargs=2, type=int, default=[], help='Create list optimizationsIds using range (-optimiztationIdRange a b: range(a, b)')

    args = parser.parse_args()

    optimizationIdList = args.optimizationIds if len(args.optimizationIds) > 0 or len(args.optimizationIdRange) != 2 else range(args.optimizationIdRange[0], args.optimizationIdRange[1])

    main(optimizationIdList, partition=args.partition, qos=args.qos, nodes=args.nodes)

