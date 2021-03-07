#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
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
        assert 'path' in jobDict and type(jobDict['path']) is str
        assert 'jobFilename' in jobDict and type(jobDict['jobFilename']) is str
        assert 'jobname' in jobDict and type(jobDict['jobname']) is str
        assert 'joboutput' in jobDict and type(jobDict['joboutput']) is str
        assert 'programm' in jobDict and type(jobDict['programm']) is str

        #Test of the optional parameter
        assert jobDict['partition'] in NeshCluster_Constants.PARTITION if 'partition' in jobDict else True
        assert jobDict['qos'] in NeshCluster_Constants.QOS if 'qos' in jobDict else True
        assert type(jobDict['nodes']) is int and jobDict['nodes'] > 0 if 'nodes' in jobDict else True
        assert type(jobDict['tasksPerNode']) is int and jobDict['tasksPerNode'] > 0 if 'tasksPerNode' in jobDict else True
        assert type(jobDict['cpusPerTask']) is int and jobDict['cpusPerTask'] > 0 if 'cpusPerTask' in jobDict else True
        assert type(jobDict['time']) is int and jobDict['time'] >= 0 if 'time' in jobDict else True
        assert type(jobDict['timeMinutes']) is int and jobDict['timeMinutes'] >= 0 and jobDict['timeMinutes'] < 60 if 'timeMinutes' in jobDict else True
        assert type(jobDict['timeSeconds']) is int and jobDict['timeSeconds'] >= 0 and jobDict['timeSeconds'] < 60 if 'timeSeconds' in jobDict else True
        assert type(jobDict['memory']) is int and jobDict['memory'] > 0 if 'memory' in jobDict else True
        assert jobDict['memoryUnit'] in NeshCluster_Constants.MEMORY_UNITS if 'memoryUnit' in jobDict else True
        assert type(jobDict['joberror']) is str if 'joberror' in jobDict else True
        assert type(jobDict['loadingModulesScript']) is str if 'loadingModulesScript' in jobDict else True
        assert type(jobDict['pythonpath']) is str if 'pythonpath' in jobDict else True

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
            runningJobs = list(self._runningJobs)
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
        os.chdir(jobDict['path'])

        x = subprocess.run(['sbatch', os.path.basename(jobDict['jobFilename'])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout_str = x.stdout.decode(encoding='UTF-8')
        matches = re.search(r'^Submitted batch job (\d+)', stdout_str)
        if matches:
            jobnum = matches.groups()[0]
            jobDict['jobnum'] = jobnum
            jobDict['finished'] = False

            self._runningJobs[jobnum] = jobDict
        else:
            #Job was not started
            logging.error('Job was not started:\n{}'.format(stdout_str))
            assert False


    def _isJobTerminated(self, jobnum):
        """
        Check, if the batch job terminated.
        @author: Markus Pfeil
        """
        assert type(jobnum) is str

        #Test if the job exists anymore
        y = subprocess.run(['squeue', '-j', jobnum], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout_str_y = y.stdout.decode(encoding='UTF-8')
        match_pos = re.search(r'^\s*JOBID\s*PARTITION\s*NAME\s*USER\s*ST\s*TIME\s*NODES\s*NODELIST\(REASON\)\s*(\d+)\s*(\w+)\s*(\w+)\s*(\w+)\s*(\w+)\s*(\d*-?\d*:?\d+:\d+)\s*(\d+)\s*(\S+)', stdout_str_y)
        match_neg_first = re.search(r'^\s*JOBID\s*PARTITION\s*NAME\s*USER\s*ST\s*TIME\s*NODES\s*NODELIST\(REASON\)\s*$', stdout_str_y)
        match_neg_second = re.search(r'^slurm_load_jobs error: Invalid job id specified', stdout_str_y)

        if (match_pos and not (match_neg_first or match_neg_second)):
            assert match_pos.groups()[0] == jobnum
            job_finished = False
        elif match_neg_first or match_neg_second:
            job_finished = True
        else:
            #Undefined output of the squeue command
            logging.error('Undefined output of the squeue command:\n{}'.format(stdout_str_y))
            assert False

        if job_finished:
            jobDict = self._runningJobs[jobnum]
            jobDict['finished'] = True
            os.chdir(jobDict['currentPath'])
            #Delete the batch job script
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

        #Quality of service
        qos = jobDict['qos'] if 'qos' in jobDict else NeshCluster_Constants.DEFAULT_QOS
        assert qos in NeshCluster_Constants.QOS

        with open(jobDict['jobFilename'], mode='w') as f:
            f.write('#!/bin/bash\n\n')

            #Job name
            f.write('#SBATCH --job-name={:s}\n'.format(jobDict['jobname']))

            #Number of nodes
            try:
                f.write('#SBATCH --nodes={:d}\n'.format(jobDict['nodes']))
            except KeyError:
                f.write('#SBATCH --nodes={:d}\n'.format(NeshCluster_Constants.DEFAULT_NODES))

            #Number of tasks per node or number of MPI processes per node
            try:
                f.write('#SBATCH --tasks-per-node={:d}\n'.format(jobDict['tasksPerNode']))
            except KeyError:
                f.write('#SBATCH --tasks-per-node={:d}\n'.format(NeshCluster_Constants.DEFAULT_TASKS_PER_NODE))

            #Number of cores per task or process
            try:
                f.write('#SBATCH --cpus-per-task={:d}\n'.format(jobDict['cpusPerTask']))
            except KeyError:
                f.write('#SBATCH --cpus-per-task={:d}\n'.format(NeshCluster_Constants.DEFAULT_CPUS_PER_TASK))

            #Real memory required per node (in gigabytes)
            memory_unit = jobDict['memoryUnit'] if 'memoryUnit' in jobDict else NeshCluster_Constants.DEFAULT_MEMORY_UNIT
            assert memory_unit in NeshCluster_Constants.MEMORY_UNITS
            try:
                f.write('#SBATCH --mem={:d}{:s}\n'.format(jobDict['memory'], memory_unit))
            except KeyError:
                f.write('#SBATCH --mem={:d}{:s}\n'.format(NeshCluster_Constants.DEFAULT_MEMORY, memory_unit))

            #Walltime in the format hours:minutes:seconds
            try:
                if ('timeMinutes' in jobDict or 'timeSeconds' in jobDict) and jobDict['time'] <= NeshCluster_Constants.WALLTIME_HOUR[qos]-1 or 'timeMinutes' not in jobDict and 'timeSeconds' not in jobDict and jobDict['time'] <= NeshCluster_Constants.WALLTIME_HOUR[qos]:
                    f.write('#SBATCH --time={:0>2d}:{:0>2d}:{:0>2d}\n'.format(jobDict['time'], jobDict['timeMinutes'] if 'timeMinutes' in jobDict else 0, jobDict['timeSeconds'] if 'timeSeconds' in jobDict else 0))
                else:
                    raise ValueError('Set walltime too big')
            except (KeyError, ValueError):
                f.write('#SBATCH --time={:0>2d}:00:00\n'.format(NeshCluster_Constants.WALLTIME_HOUR[qos]))

            #Stdout file
            f.write('#SBATCH --output={:s}\n'.format(jobDict['joboutput']))

            #Stderr file; if not specifed, stderr is redirected to stdout file
            try:
                f.write('#SBATCH --error={:s}\n'.format(jobDict['joberror']))
            except KeyError:
                pass

            #Slum partition (Batch class)
            try:
                f.write('#SBATCH --partition={:s}\n'.format(jobDict['partition']))
            except KeyError:
                f.write('#SBATCH --partition={:s}\n'.format(NeshCluster_Constants.DEFAULT_PARTITION))

            #Define a quality of service
            try:
                f.write('#SBATCH --qos={:s}\n\n'.format(jobDict['qos']))
            except KeyError:
                f.write('#SBATCH --qos={:s}\n\n'.format(NeshCluster_Constants.DEFAULT_QOS))

            #Environment variable for the number of computational cores (threads)
            try:
                f.write('export OMP_NUM_THREADS={:d}\n\n'.format(jobDict['cpusPerTask']))
            except KeyError:
                f.write('export OMP_NUM_THREADS={:d}\n\n'.format(NeshCluster_Constants.DEFAULT_CPUS_PER_TASK))

            #Script loading modules
            try:
                f.write('source {:s}\n\n'.format(jobDict['loadingModulesScript']))
            except KeyError:
                pass

            #Environment variable for Python (list of directories that Python add to the sys.path directory list)
            try:
                f.write('export PYTHONPATH={}\n\n'.format(jobDict['pythonpath']))
            except KeyError:
                pass
            
            #Run python program
            f.write('python3 {}\n\n'.format(jobDict['programm']))

            #Summary of resources that have been consumed by the batch calculation
            f.write('jobinfo\n')

