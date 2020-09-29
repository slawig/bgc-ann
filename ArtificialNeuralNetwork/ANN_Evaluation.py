#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import subprocess
import re
import multiprocessing as mp
import time

import ann.network.constants as ANN_Constants
import neshCluster.constants as NeshCluster_Constants
import ann.evaluation.constants as Evaluation_Constants


def main(annIdList=[222, 213, 207], parameterIdList=range(0, ANN_Constants.PARAMETERID_MAX_TEST+1), parallelJobs=20, queue='clmedium', cores=2):
    """
    Evaluate the artificial neural networks using
      - the prediction
      - the mass-corrected prediction
      - the prediction as intial concentration for a spin up over 1000 model years
      - the mass-corrected prediction as intial concentration for a spin up over 1000 model years
      - the prediction as intial concentration for a spin up with tolerance 10**(-4)
      - the mass-corrected prediction as intial concentration for a spin up with tolerance 10**(-4)
    @author: Markus Pfeil
    """
    assert type(annIdList) is list
    assert type(parameterIdList) is list
    assert type(parallelJobs) is int and 0 < parallelJobs
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and 0 < cores

    joblist = []
    #Calculate the spin up with tolerance 10**(-4) using constant initial concentration
    joblist = joblist + generateJoblist(annIdList=annIdList, parameterIdList=parameterIdList, massAdjustment=False, tolerance=10**(-4), spinupToleranceReference=True, queue=queue, cores=cores)

    #Calculate the spin up over 1000 model years using the prediction as intial concentration
    joblist = joblist + generateJoblist(annIdList=annIdList, parameterIdList=parameterIdList, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue=queue, cores=cores)

    #Calculate the spin up over 1000 model years using the mass-corrected prediction as intial concentration
    joblist = joblist + generateJoblist(annIdList=annIdList, parameterIdList=parameterIdList, massAdjustment=True, tolerance=None, spinupToleranceReference=False, queue=queue, cores=cores)
    
    #Calculate the spin up with a tolerance of 10**(-4) using the prediction as intial concentration
    joblist = joblist + generateJoblist(annIdList=annIdList, parameterIdList=parameterIdList, massAdjustment=False, tolerance=10**(-4), spinupToleranceReference=False, queue=queue, cores=cores)

    #Calculate the spin up over 1000 model years using the mass-corrected prediction as intial concentration
    joblist = joblist + generateJoblist(annIdList=annIdList, parameterIdList=parameterIdList, massAdjustment=True, tolerance=10**(-4), spinupToleranceReference=False, queue=queue, cores=cores)
    
    #
    with mp.Pool(parallelJobs) as p:
        jobnum = p.starmap(runJob, joblist)


def checkJob(annId, parameterId, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2):
    """
    Check, if the job output exists for the given artificial neural network and parameterId.
    @author: Markus Pfeil
    """
    assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
    assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or type(tolerance) is float and 0 < tolerance
    assert type(spinupToleranceReference) is bool
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and 0 < cores
    
    annDb = Ann_Database()
    annType, model = annDb.get_annTypeModel(annId)
    annDb.close_connection()
    if spinupToleranceReference:
        joboutputFile = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Logfile', Evaluation_Constants.PATTERN_LOGFILE_SPINUP_REFERENCE.format(model, parameterId, 0.0 if tolerance is None else tolerance, cores * NeshCluster_Constants.CPUNUM[queue]))
    else:
        joboutputFile = os.path.join(ANN_Constants.PATH, 'Prediction', 'Logfile', Evaluation_Constants.PATTERN_LOGFILE.format(annType, model, annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, cores * NeshCluster_Constants.CPUNUM[queue]))

    return os.path.exists(joboutputFile) and os.path.isfile(joboutputFile)


def generateJoblist(annIdList=range(0, ANN_Constants.ANNID_MAX+1), parameterIdList=range(0, ANN_Constants.PARAMETERID_MAX_TEST+1), massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2):
    """
    Create a list of jobs for the given artificial neural network and parameter indices
    @author: Markus Pfeil
    """
    assert type(annIdList) is list
    assert type(parameterIdList) is list
    assert type(massAdjustment) is bool
    assert tolerance is None or type(tolerance) is float and 0 < tolerance
    assert type(spinupToleranceReference) is bool
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and 0 < cores:q
    
    joblist = []

    for annId in annIdList:
        assert type(annId) is int and annId in range(0, ANN_Constants.ANNID_MAX+1)
        for parameterId in parameterIdList:
            assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
            if not checkJob(annId, parameterId, massAdjustment=massAdjustment, tolerance=tolerance, spinupToleranceReference=spinupToleranceReference, queue=queue, cores=cores):
                joblist.append([annId, parameterId, massAdjustment, tolerance, spinupToleranceReference, queue, cores])
    return joblist


def runJob(annId, parameterId, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2):
    """
    Start the job for the given artificial neural networt and the parameter id using the given queue and cores
    @author: Markus Pfeil
    """
    assert annId in range(0,ANN_Constants.ANNID_MAX+1)
    assert parameterId in range(0,ANN_Constants.PARAMETERID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or type(tolerance) is float and 0 < tolerance
    assert type(spinupToleranceReference) is bool
    assert queue in NeshCluster_Constants.QUEUE
    assert cores > 0

    jobfile = writeJobfile(annId, parameterId, massAdjustment=massAdjustment, tolerance=tolerance, spinupToleranceReference=spinupToleranceReference, queue=queue, cores=cores)

    akt_path = os.getcwd()
    os.chdir(os.path.dirname(jobfile))

    x = subprocess.run(['qsub', os.path.basename(jobfile)], stdout=subprocess.PIPE)

    stdout_str = x.stdout.decode(encoding='UTF-8')
    matches = re.search(r'^Request (\d+).nesh-batch submitted to queue: (cl\w+)', stdout_str)
    if matches:
        [jobnum, jobqueue] = matches.groups()
    else:
        jobnum = '0'

    #Wait for the existing job
    job_exist = True
    while (job_exist):
        y = subprocess.run(['qstat', jobnum], stdout=subprocess.PIPE)
        #Test, if the job exists anymore
        stdout_str_y = y.stdout.decode(encoding='UTF-8')
        match_neg = re.search(r'Batch Request: (\d+).nesh-batch does not exist on nesh-batch', stdout_str_y)
        match_pos = re.search(r'(\d+).nesh-batc?h?\s? \S+\s+sunip350', stdout_str_y)

        if (match_pos and (not match_neg)):
            time.sleep(120)
        else:
            job_exist = False

    #Delete the batch job script
    os.chdir(akt_path)
    os.remove(jobfile)


def writeJobfile(annId, parameterId, massAdjustment=False, tolerance=None, spinupToleranceReference=False, queue='clmedium', cores=2):
    """
    Write jobfile for the nesh cluster
    @author: Markus Pfeil
    """
    assert annId in range(0,ANN_Constants.ANNID_MAX+1)
    assert parameterId in range(0,ANN_Constants.PARAMETERID_MAX+1)
    assert type(massAdjustment) is bool
    assert tolerance is None or type(tolerance) is float and 0 < tolerance
    assert type(spinupToleranceReference) is bool
    assert queue in NeshCluster_Constants.QUEUE
    assert cores > 0

    annDb = Ann_Database()
    annType, model = annDb.get_annTypeModel(annId)
    annDb.close_connection()
    if spinupToleranceReference:
        filename = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Jobfile', Evaluation_Constants.PATTERN_JOBFILE_SPINUP_REFERENCE.format(model, parameterId, 0.0 if tolerance is None else tolerance, cores*NeshCluster_Constants.CPUNUM[queue]))
        joboutput = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT_SPINUP_REFERENCE.format(model, parameterId, 0.0 if tolerance is None else tolerance, cores*NeshCluster_Constants.CPUNUM[queue]))
    else:
        filename = os.path.join(ANN_Constants.PATH, 'Prediction', 'Jobfile', Evaluation_Constants.PATTERN_JOBFILE.format(annType, model, annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, cores*NeshCluster_Constants.CPUNUM[queue]))
        joboutput = os.path.join(ANN_Constants.PATH, 'Prediction', 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT.format(annType, model, annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, cores*NeshCluster_Constants.CPUNUM[queue]))

    f = open(filename, mode='w')

    f.write('#!/bin/bash\n\n')
    f.write('#PBS -N ANN_{}_{}_Prediction\n'.format(model, annId))
    f.write('#PBS -j o\n')
    f.write('#PBS -o {}\n'.format(joboutput))
    f.write('#PBS -b {:d}\n'.format(cores))
    f.write('#PBS -l cpunum_job={:d}\n'.format(NeshCluster_Constants.CPUNUM[queue]))
    f.write('#PBS -l elapstim_req={:d}:00:00\n'.format(NeshCluster_Constants.ELAPSTIM[queue]))
    f.write('#PBS -l memsz_job=4gb\n')
    f.write('#PBS -T intmpi\n')
    f.write('#PBS -q {}\n'.format(queue))
    f.write('cd $PBS_O_WORKDIR\n')
    f.write('#\n')
    f.write('qstat -l -F ehost ${PBS_JOBID/0:}\n')
    f.write('#\n')
    f.write('. /sfs/fs2/work-sh1/sunip350/metos3d/nesh_metos3d_setup_v0.6.9.sh\n')
    f.write('#\n')
    f.write('export PYTHONPATH=/sfs/fs2/work-sh1/sunip350/python/util:/sfs/fs2/work-sh1/sunip350/python/ann:/sfs/fs2/work-sh1/sunip350/python/ArtificialNeuralNetwork\n')
    f.write('#\n')
    f.write('python3 /sfs/fs2/work-sh1/sunip350/python/ArtificialNeuralNetwork/ANN_EvaluationJob.py {:d} {:d} -queue {:d} -cores {:d}'.format(annId, parameterId, queue, cores))

    #Optional Parameter
    if tolerance is not None:
        f.write(' -tolerance {:f}'.format(tolerance))

    if massAdjustment:
        f.write(' --massAdjustment')

    if spinupToleranceReference:
        f.write(' --spinupToleranceReference')

    f.write('\n')

    f.write('export TMPDIR="/scratch/"`echo $PBS_JOBID | cut -f2 -d\:`\n')

    f.close()

    return filename



if __name__ == '__main__':
    main()
