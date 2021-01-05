#!/usr/bin/env python
# -*- coding: utf8 -*

import os

import neshCluster.constants as NeshCluster_Constants
from neshCluster.JobAdministration import JobAdministration
from database.SBO_Database import SBO_Database
import sbo.constants as SBO_Constants


def main(optimizationIdList, queue=NeshCluster_Constants.DEFAULT_QUEUE, cores=NeshCluster_Constants.DEFAULT_CORES):
    """
    Run the surrogate based optimization for the given optimizationIds.
    @author: Markus Pfeil
    """
    assert type(optimizationIdList) in [list, range]
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and 0 < cores

    sbo = SurrogateBasedOptimizationJobAdministration(optimizationIdList=optimizationIdList, queue=queue, cores=cores)
    sbo.generateJobList()
    sbo.runJobs()



class SurrogateBasedOptimizationJobAdministration(JobAdministration):
    """
    Class for the administration of the jobs organizing the run of the surrogate based optimizations.
    @author: Markus Pfeil
    """

    def __init__(self, optimizationIdList, queue=NeshCluster_Constants.DEFAULT_QUEUE, cores=NeshCluster_Constants.DEFAULT_CORES):
        """
        Initialisation of the evaluation jobs of the ANN with the given annId.
        @author: Markus Pfeil
        """
        assert type(optimizationIdList) in [list, range]
        assert queue in NeshCluster_Constants.QUEUE
        assert type(cores) is int and 0 < cores

        JobAdministration.__init__(self)

        self._optimizationIdList = optimizationIdList
        self._queue = queue
        self._cores = cores


    def generateJobList(self):
        """
        Create a list of jobs to run the surrogate based optimization.
        @author: Markus Pfeil
        """
        self._sboDb = SBO_Database()

        for optimizationId in self._optimizationIdList:
            if not self._checkJob(optimizationId):
                programm = 'SBO_Jobcontrol.py {:d} -queue {:s} -cores {:d}'.format(optimizationId, self._queue, self._cores)

                jobDict = {}
                jobDict['jobFilename'] = os.path.join(SBO_Constants.PATH, 'Optimization', 'Jobfile', SBO_Constants.PATTERN_JOBFILE.format(optimizationId))
                jobDict['jobname'] = 'SBO_{:d}'.format(optimizationId)
                jobDict['joboutput'] = os.path.join(SBO_Constants.PATH, 'Optimization', 'Logfile', SBO_Constants.PATTERN_JOBOUTPUT.format(optimizationId))
                jobDict['programm'] = os.path.join(SBO_Constants.PROGRAMM_PATH, programm)
                jobDict['queue'] = self._queue
                jobDict['cores'] = self._cores
                jobDict['memory'] = 48

                self.addJob(jobDict)

        self._sboDb.close_connection()


    def _checkJob(self, optimizationId):
        """
        Check, if the output exists for the optimization with the given optimizationId
        @author: Markus Pfeil
        """
        assert type(optimizationId) is int and 0 <= optimizationId

        existsOptimizationId = self._sboDb.exists_optimization(optimizationId)
        existsIteration = self._sboDb.exists_iteration(optimizationId)
        sboPath = os.path.join(SBO_Constants.PATH, 'Optimization', SBO_Constants.PATH_OPTIMIZATION.format(optimizationId))

        return existsOptimizationId and existsIteration and os.path.exists(sboPath) and os.path.isdir(sboPath)




if __name__ == '__main__':
    optimizationIdList = [48, 136, 160, 168, 188, 189, 205] #range(338, 390)
    main(optimizationIdList)

