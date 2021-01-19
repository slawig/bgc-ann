#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import subprocess


class JobAdministration():
    """
    Class for the administration of jobs running on a standalone computer.
    @author: Markus Pfeil
    """

    def __init__(self):
        """
        Initialisation of the job administration class.
        @author: Markus Pfeil
        """
        self._jobList = []
        self._runningJobs = {}


    def addJob(self, jobDict):
        """
        Add a job to the job liste in order to run this job.
        @author: Markus Pfeil
        """
        assert type(jobDict) is dict
        assert 'path' in jobDict
        assert 'programm' in jobDict
        assert 'joboutput' in jobDict

        self._jobList.append(jobDict)


    def runJobs(self):
        """
        Start the jobs in the job list on the standalone computer.
        @author: Markus Pfeil
        """
        currentPath = os.getcwd()
        
        for jobDict in self._jobList:
            self._startJob(jobDict) 
            self._evaluateResult(jobDict)

        os.chdir(currentPath)


    def _startJob(self, jobDict):
        """
        Start the job.
        @author: Markus Pfeil
        """
        assert type(jobDict) is dict

        os.chdir(jobDict['path'])
        x = subprocess.run(['python3'] + [a for a in jobDict['programm'].split(' ')], stdout=subprocess.PIPE)
        with open(jobDict['joboutput'], mode='w', encoding='utf-8') as fid:
            fid.write(x.stdout.decode(encoding='UTF-8'))


    def _evaluateResult(self, jobDict):
        """
        Evaluate the result of the job.
        This function have to be implemented in every special case.
        @author: Markus Pfeil
        """
        return True

