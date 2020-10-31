#!/usr/bin/env python
# -*- coding: utf8 -*

import os

import ann.network.constants as ANN_Constants
import neshCluster.constants as NeshCluster_Constants
import ann.evaluation.constants as Evaluation_Constants
from neshCluster.JobAdministration import JobAdministration
from ann.database.access import Ann_Database


def main(annIdList=[222, 213, 207], parameterIdList=range(0, ANN_Constants.PARAMETERID_MAX_TEST+1), queue='clmedium', cores=2):
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
    assert type(parameterIdList) in [list, range]
    assert queue in NeshCluster_Constants.QUEUE
    assert type(cores) is int and 0 < cores

    for annId in annIdList:
        evaluation = ANN_Evaluation(annId, parameterIdList=parameterIdList, queue=queue, cores=cores)
        evaluation.generateJobList()

    evaluation.runJobs()



class ANN_Evaluation(JobAdministration):
    """
    Class for the administration of the jobs organizing the evaluation of an ANN.
    @author: Markus Pfeil
    """

    def __init__(self, annId, parameterIdList=range(0, ANN_Constants.PARAMETERID_MAX_TEST+1), queue='clmedium', cores=2, trajectoryFlag=True):
        """
        Initialisation of the evaluation jobs of the ANN with the given annId.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert type(parameterIdList) in [list, range]
        assert queue in NeshCluster_Constants.QUEUE
        assert type(cores) is int and 0 < cores
        assert type(trajectoryFlag) is bool

        JobAdministration.__init__(self)

        self._annId = annId
        self._parameterIdList = parameterIdList
        self._queue = queue
        self._cores = cores
        self._trajectoryFlag = trajectoryFlag

        annDb = Ann_Database()
        self._annType, self._model = annDb.get_annTypeModel(self._annId)
        annDb.close_connection()


    def generateJobList(self):
        """
        Create a list of jobs to evaluate the approximation using the given artificial neural network.
        @author: Markus Pfeil
        """
        #Calculate the spin up with tolerance 10**(-4) using constant initial concentration
        self._addEvaluationJobs(massAdjustment=False, tolerance=10**(-4), spinupToleranceReference=True)

        #Calculate the spin up over 1000 model years using the prediction as intial concentration
        self._addEvaluationJobs(massAdjustment=False, tolerance=None)

        #Calculate the spin up over 1000 model years using the mass-corrected prediction as intial concentration
        self._addEvaluationJobs(massAdjustment=True, tolerance=None)

        #Calculate the spin up with a tolerance of 10**(-4) using the prediction as intial concentration
        self._addEvaluationJobs(massAdjustment=False, tolerance=10**(-4))

        #Calculate the spin up over 1000 model years using the mass-corrected prediction as intial concentration
        self._addEvaluationJobs(massAdjustment=True, tolerance=10**(-4))


    def _addEvaluationJobs(self, massAdjustment=False, tolerance=None, spinupToleranceReference=False):
        """
        Create a list of jobs to evaluate the approximation using the given artificial neural network for the list of parameter indices.
        @author: Markus Pfeil
        """
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and 0 < tolerance
        assert type(spinupToleranceReference) is bool

        for parameterId in self._parameterIdList:
            assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
            if not self._checkJob(parameterId, massAdjustment=massAdjustment, tolerance=tolerance, spinupToleranceReference=spinupToleranceReference):

                if spinupToleranceReference:
                    filename = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Jobfile', Evaluation_Constants.PATTERN_JOBFILE_SPINUP_REFERENCE.format(self._model, parameterId, 0.0 if tolerance is None else tolerance, self._cores * NeshCluster_Constants.CPUNUM[self._queue]))
                    joboutput = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT_SPINUP_REFERENCE.format(self._model, parameterId, 0.0 if tolerance is None else tolerance, self._cores * NeshCluster_Constants.CPUNUM[self._queue]))
                else:
                    filename = os.path.join(ANN_Constants.PATH, 'Prediction', 'Jobfile', Evaluation_Constants.PATTERN_JOBFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, self._cores * NeshCluster_Constants.CPUNUM[self._queue]))
                    joboutput = os.path.join(ANN_Constants.PATH, 'Prediction', 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT.format(self._annType, self._model, self._annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, self._cores * NeshCluster_Constants.CPUNUM[self._queue]))

                programm = 'ANN_EvaluationJob.py {:d} {:d} -queue {:s} -cores {:d}'.format(self._annId, parameterId, self._queue, self._cores)
                #Optional Parameter
                if tolerance is not None:
                    programm = programm + ' -tolerance {:f}'.format(tolerance)
                if massAdjustment:
                    programm = programm + ' --massAdjustment'
                if spinupToleranceReference:
                    programm = programm + ' --spinupToleranceReference'
                if self._trajectoryFlag:
                    programm = programm + ' --trajectory'

                jobDict = {}
                jobDict['jobFilename'] = filename
                jobDict['jobname'] = 'ANN_{}_{}_Prediction'.format(self._model, self._annId)
                jobDict['joboutput'] = joboutput
                jobDict['programm'] = os.path.join(NeshCluster_Constants.PROGRAMM_PATH, programm)
                jobDict['queue'] = self._queue
                jobDict['cores'] = self._cores

                self.addJob(jobDict)


    def _checkJob(self, parameterId, massAdjustment=False, tolerance=None, spinupToleranceReference=False):
        """
        Check, if the job output exists for the given artificial neural network and parameterId.
        @author: Markus Pfeil
        """
        assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and 0 < tolerance
        assert type(spinupToleranceReference) is bool

        if spinupToleranceReference:
            joboutputFile = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Logfile', Evaluation_Constants.PATTERN_LOGFILE_SPINUP_REFERENCE.format(self._model, parameterId, 0.0 if tolerance is None else tolerance, self._cores * NeshCluster_Constants.CPUNUM[self._queue]))
        else:
            joboutputFile = os.path.join(ANN_Constants.PATH, 'Prediction', 'Logfile', Evaluation_Constants.PATTERN_LOGFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, self._cores * NeshCluster_Constants.CPUNUM[self._queue]))

        return os.path.exists(joboutputFile) and os.path.isfile(joboutputFile)




if __name__ == '__main__':
    main()

