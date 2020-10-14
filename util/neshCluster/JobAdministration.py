#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import subprocess
import re
import time

import neshCluster.constants as NeshCluster_Constants


class JobAdministration():
    """
    Class for the administration of job running on the NEC HPC-Linux-Cluster of the CAU Kiel.
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
        assert 'jobFilename' in jobDict
        assert 'jobname' in jobDict
        assert 'joboutput' in jobDict
        assert 'programm' in jobDict

        #Set optional parameter        
        if not 'queue' in jobDict:
            jobDict['queue'] = NeshCluster_Constants.DEFAULT_QUEUE
        else:
            assert jobDict['queue'] in NeshCluster_Constants.QUEUE

        if not 'cores' in jobDict:
            jobDict['cores'] = NeshCluster_Constants.DEFAULT_CORES
        else:
            assert type(jobDict['cores']) is int and 0 < jobDict['cores']
        
        if not 'memory' in jobDict:
             jobDict['memory'] = NeshCluster_Constants.DEFAULT_MEMORY
        else:
            assert type(jobDict['memory']) is int and 0 < jobDict['memory']

        if not 'pythonpath' in jobDict:
            jobDict['pythonpath'] = NeshCluster_Constants.DEFAULT_PYTHONPATH
        
        self._jobList.append(jobDict)


    def runJobs(self):
        """
        Start the jobs in the job list on the NEC HPC-Linux-Cluster.
        @author: Markus Pfeil
        """
        jobsToStart = self._jobList.copy()
        jobsToStart.reverse()
        
        while (len(self._runningJobs) > 0 or len(jobsToStart) > 0):
            #Start jobs
            while (len(self._runningJobs) < NeshCluster_Constants.PARALLEL_JOBS and len(jobsToStart) > 0):
                self._startJob(jobsToStart.pop())

            #Check running Jobs
            runningJobs = self._runningJobs
            for jobnum in runningJobs:
                if self._isJobTerminated(jobnum):
                    del self._runningJobs[jobnum]

            #Wait for the next check
            time.sleep(NeshCluster_Constants.TIME_SLEEP)
            

    def _startJob(self, jobDict):
        """
        Start the job.
        @author: Markus Pfeil
        """
        assert type(jobDict) is dict

        self._writeJobfile(jobDict)
        jobDict['currentPath'] = os.getcwd()
        os.chdir(os.path.dirname(jobDict['jobFilename']))

        x = subprocess.run(['qsub', os.path.basename(jobDict['jobFilename'])], stdout=subprocess.PIPE)
        stdout_str = x.stdout.decode(encoding='UTF-8')
        matches = re.search(r'^Request (\d+).nesh-batch submitted to queue: (cl\w+)', stdout_str)
        if matches:
            [jobnum, jobqueue] = matches.groups()
            jobDict['jobnum'] = jobnum
            jobDict['finished'] = False

            self._runningJobs[jobnum] = jobDict
        else:
            #Job was not started
            assert False


    def _isJobTerminated(self, jobnum):
        """
        Check, if the batch job terminated.
        @author: Markus Pfeil
        """
        assert type(jobnum) is str

        #Test if the job exists anymore
        y = subprocess.run(['qstat', jobnum], stdout=subprocess.PIPE)
        stdout_str_y = y.stdout.decode(encoding='UTF-8')
        match_neg = re.search(r'Batch Request: (\d+).nesh-batch does not exist on nesh-batch', stdout_str_y)
        match_pos = re.search(r'(\d+).nesh-batc?h?\s? \S+\s+sunip350', stdout_str_y)

        if (match_pos and (not match_neg)):
            job_finished = False
        else:
            job_finished = True

        if job_finished:
            #Delete the batch job script
            jobDict = self._runningJobs[jobnum]
            jobDict['finished'] = True
            os.chdir(jobDict['currentPath'])
            os.remove(jobDict['jobFilename'])

            job_finished = self._evaluateResult(jobDict)

        return job_finished

    
    def _evaluateResult(self, jobDict):
        """
        Evaluate the result of the job.
        This function have to be implemented in every special case.
        @author: Markus Pfeil
        """
        return True
        

    def _writeJobfile(self, jobDict): 
        """
        Write jobfile for the NEC HPC-Linux-Cluster of the CAU Kiel.
        @author: Markus Pfeil
        """
        assert type(jobDict) is dict
        assert not os.path.exists(jobDict['jobFilename'])

        with open(jobDict['jobFilename'], mode='w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('#PBS -N {:s}\n'.format(jobDict['jobname']))
            f.write('#PBS -j o\n')
            f.write('#PBS -o {:s}\n'.format(jobDict['joboutput']))
            f.write('#PBS -b {:d}\n'.format(jobDict['cores']))
            f.write('#PBS -l cpunum_job={:d}\n'.format(NeshCluster_Constants.CPUNUM[jobDict['queue']]))
            f.write('#PBS -l elapstim_req={:d}:00:00\n'.format(NeshCluster_Constants.ELAPSTIM[jobDict['queue']]))
            f.write('#PBS -l memsz_job={:d}gb\n'.format(jobDict['memory']))
            f.write('#PBS -T intmpi\n')
            f.write('#PBS -q {}\n'.format(jobDict['queue']))
            f.write('cd $PBS_O_WORKDIR\n')
            f.write('#\n')
            f.write('qstat -l -F ehost ${PBS_JOBID/0:}\n')
            f.write('#\n')
            f.write('. /sfs/fs2/work-sh1/sunip350/metos3d/nesh_metos3d_setup_v0.6.9.sh\n')
            f.write('#\n')
            f.write('export PYTHONPATH={}\n'.format(jobDict['pythonpath']))
            f.write('#\n')
            f.write('python3 {}\n'.format(jobDict['programm']))
            f.write('\n')
            f.write('export TMPDIR="/scratch/"`echo $PBS_JOBID | cut -f2 -d\:`\n')

