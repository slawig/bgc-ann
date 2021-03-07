#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os

import ann.network.constants as ANN_Constants
import neshCluster.constants as NeshCluster_Constants
import ann.evaluation.constants as Evaluation_Constants
from ann.database.access import Ann_Database
from system.system import SYSTEM
if SYSTEM == 'PC':
    from standaloneComputer.JobAdministration import JobAdministration
else:
    from neshCluster.JobAdministration import JobAdministration


def main(partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None):
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
    assert partition in NeshCluster_Constants.PARTITION
    assert qos in NeshCluster_Constants.QOS
    assert type(nodes) is int and 0 < nodes
    assert memory is None or type(memory) is int and 0 < memory
    assert time is None or type(time) is int and 0 < time

    evaluation = ANN_Evaluation(partition=partition, qos=qos, nodes=nodes, memory=memory, time=time)
    evaluation.generateJobList()
    evaluation.runJobs()



class ANN_Evaluation(JobAdministration):
    """
    Class for the administration of the jobs organizing the evaluation of an ANN.
    @author: Markus Pfeil
    """

    def __init__(self, partition=NeshCluster_Constants.DEFAULT_PARTITION, qos=NeshCluster_Constants.DEFAULT_QOS, nodes=NeshCluster_Constants.DEFAULT_NODES, memory=None, time=None, trajectoryFlag=True):
        """
        Initialisation of the evaluation jobs of the ANN with the given annId.
        @author: Markus Pfeil
        """
        assert partition in NeshCluster_Constants.PARTITION
        assert qos in NeshCluster_Constants.QOS
        assert type(nodes) is int and 0 < nodes
        assert memory is None or type(memory) is int and 0 < memory
        assert time is None or type(time) is int and 0 < time
        assert type(trajectoryFlag) is bool

        JobAdministration.__init__(self)

        self._annId = 221  #TODO Set explicitly the annId
        self._partition = partition
        self._qos = qos
        self._nodes = nodes
        self._memory = memory
        self._time = time
        self._trajectoryFlag = trajectoryFlag

        annDb = Ann_Database()
        self._annType, self._model = annDb.get_annTypeModel(self._annId)
        annDb.close_connection()


    def generateJobList(self):
        """
        Create a list of jobs to evaluate the approximation using the given artificial neural network.
        @author: Markus Pfeil
        """
        self._addEvaluationJobs(parameterId=39, massAdjustment=False, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=40, massAdjustment=True, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=42, massAdjustment=False, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=43, massAdjustment=True, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=46, massAdjustment=True, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=56, massAdjustment=False, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=59, massAdjustment=False, tolerance=10**(-4))
        self._addEvaluationJobs(parameterId=62, massAdjustment=False, tolerance=10**(-4))


    def _addEvaluationJobs(self, parameterId=0, massAdjustment=False, tolerance=None, spinupToleranceReference=False):
        """
        Create a list of jobs to evaluate the approximation using the given artificial neural network for the list of parameter indices.
        @author: Markus Pfeil
        """
        assert type(parameterId) is int and parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and 0 < tolerance
        assert type(spinupToleranceReference) is bool

        if spinupToleranceReference:
            filename = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Jobfile', Evaluation_Constants.PATTERN_JOBFILE_SPINUP_REFERENCE.format(self._model, parameterId, 0.0 if tolerance is None else tolerance, self._nodes * NeshCluster_Constants.CORES))
            joboutput = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT_SPINUP_REFERENCE.format(self._model, parameterId, 0.0 if tolerance is None else tolerance, self._nodes * NeshCluster_Constants.CORES))
        else:
            filename = os.path.join(ANN_Constants.PATH, 'Prediction', 'Jobfile', Evaluation_Constants.PATTERN_JOBFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, self._nodes * NeshCluster_Constants.CORES))
            joboutput = os.path.join(ANN_Constants.PATH, 'Prediction', 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT.format(self._annType, self._model, self._annId, parameterId, massAdjustment, 0.0 if tolerance is None else tolerance, self._nodes * NeshCluster_Constants.CORES))

        programm = 'ANN_EvaluationJob.py {:d} {:d} -nodes {:d}'.format(self._annId, parameterId, self._nodes)
        #Optional Parameter
        if tolerance is not None:
            programm = programm + ' -tolerance {:f}'.format(tolerance)
        if massAdjustment:
            programm = programm + ' --massAdjustment'
        if spinupToleranceReference:
            programm = programm + ' --spinupToleranceReference'
        if self._trajectoryFlag:
            programm = programm + ' --trajectory'
        programm = programm + ' --evaluation'

        jobDict = {}
        jobDict['jobFilename'] = filename
        jobDict['path'] = os.path.dirname(filename)
        jobDict['jobname'] = 'ANN_{}_{}_Prediction'.format(self._model, self._annId)
        jobDict['joboutput'] = joboutput
        jobDict['programm'] = os.path.join(NeshCluster_Constants.PROGRAMM_PATH, programm)
        jobDict['partition'] = self._partition
        jobDict['qos'] = self._qos
        jobDict['nodes'] = self._nodes
        jobDict['pythonpath'] = NeshCluster_Constants.DEFAULT_PYTHONPATH
        jobDict['loadingModulesScript'] = NeshCluster_Constants.DEFAULT_LOADING_MODULES_SCRIPT

        if self._memory is not None:
            jobDict['memory'] = self._memory
        if self._time is not None:
            jobDict['time'] = self._time

        self.addJob(jobDict)



if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('-partition', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_PARTITION, default=NeshCluster_Constants.DEFAULT_PARTITION, help='Partition of slum on the Nesh-Cluster (Batch class)')
    parser.add_argument('-qos', nargs='?', type=str, const=NeshCluster_Constants.DEFAULT_QOS, default=NeshCluster_Constants.DEFAULT_QOS, help='Quality of service on the Nesh-Cluster')
    parser.add_argument('-nodes', nargs='?', type=int, const=NeshCluster_Constants.DEFAULT_NODES, default=NeshCluster_Constants.DEFAULT_NODES, help='Number of nodes for the job on the Nesh-Cluster')
    parser.add_argument('-memory', nargs='?', type=int, const=None, default=None, help='Memory in GB for the job on the Nesh-Cluster')
    parser.add_argument('-time', nargs='?', type=int, const=None, default=None, help='Time in hours for the job on the Nesh-Cluster')

    args = parser.parse_args()

    main(partition=args.partition, qos=args.qos, nodes=args.nodes, memory=args.memory, time=args.time)

