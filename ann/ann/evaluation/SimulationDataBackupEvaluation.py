#!/usr/bin/env python
# -*- coding: utf8 -*

import bz2
import logging
import os
import shutil
import tarfile

import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.metos3d.Metos3d as Metos3d
import ann.network.constants as ANN_Constants
import ann.evaluation.constants as Evaluation_Constants
from ann.evaluation.AbstractClassData import AbstractClassData
from ann.geneticAlgorithm.geneticAlgorithm import GeneticAlgorithm


class SimulationDataBackupEvaluation(AbstractClassData):
    """
    Create a backup of the simulation data using an artificial neural network to approximate a steady annual cycle.
    @author: Markus Pfeil
    """

    def __init__(self, annId):
        """
        Constructor of the class for generation of the backup.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        AbstractClassData.__init__(self, annId)

        #Path
        self._pathTarfile = self._pathPrediction

        self._getAnnFilenameList()

        #TODO Wie kann dieses Konzept verbessert werden? Neue Evaluations enthalten alle vier Optionen
        self._massAdjustmentList = [38, 56, 122, 170, 200, 201, 202, 203, 204, 205, 206, 207, 208, 213, 214, 219, 220, 221, 222, 225, 226, 228, 231, 235, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247]
        self._toleranceList = [10**(-4)]
        self._spinupToleranceDic = {10**(-4): [38, 56, 122, 170, 200, 201, 202, 203, 204, 205, 206, 207, 208, 213, 214, 219, 220, 221, 222, 225, 226, 228, 231, 235, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247]}
        self._spinupToleranceDicNoMassAdjustment = {10**(-4): [208, 219, 220, 221, 222, 225, 226, 228, 231, 235, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247]}

        logging.info('***Initialization of SimulationDataBackupEvaluation:***\nANN: {}\nAnnId: {:d}\nModel: {}'.format(self._annType, self._annId, self._model))


    def _getAnnFilenameList(self):
        """
        Set a list with the filenames including the path of the neural network
        @author: Markus Pfeil
        """
        self._annFilenameList = []
        if self._annType in ['fcn', 'cnn']:
            self._annFilenameList.append(os.path.join(ANN_Constants.PATH_FCN, self._model, ANN_Constants.FCN.format(self._annNumber), ANN_Constants.ANN_FILENAME_FCN.format(self._annNumber)))
            self._annFilenameList.append(os.path.join(ANN_Constants.PATH_FCN, self._model, ANN_Constants.FCN.format(self._annNumber), ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(self._annNumber)))
        elif self._annType == 'set':
            self._annFilenameList.append(os.path.join(ANN_Constants.PATH_SET, ANN_Constants.FCN_SET.format(self._annNumber), ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(self._annNumber)))
            self._annFilenameList.append(os.path.join(ANN_Constants.PATH_SET, ANN_Constants.FCN_SET.format(self._annNumber), ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(self._annNumber)))
            self._annFilenameList.append(os.path.join(ANN_Constants.PATH_SET, ANN_Constants.FCN_SET.format(self._annNumber), ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(self._annNumber)))
        elif self._annType == 'setgen':
            geneticAlgorithm = GeneticAlgorithm(gid=self._annNumber)
            (gen, uid, ann_path) = geneticAlgorithm.readBestGenomeFile()
            self._annFilenameList.append(os.path.join(ann_path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(uid)))
            self._annFilenameList.append(os.path.join(ann_path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(uid)))
            self._annFilenameList.append(os.path.join(ann_path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(uid)))
        else:
            assert False


    def _generateSimPathPostfix(self, parameterId):
        """
        Generate the path for all different kinds of simulations for the given parameterId
        @author: Markus Pfeil
        """
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX + 1)

        simulationPath = []

        #Simulation path for the evaluation without adjustment of the mass
        simulationPath.append((False, None, os.path.join('Parameter_{:0>3d}'.format(parameterId))))

        #Simulation path for the evaluation using adjustment of the mass
        if self._annId in self._massAdjustmentList:
            simulationPath.append((True, None, os.path.join('Parameter_{:0>3d}'.format(parameterId), 'MassAdjustment')))

        #Simulation path for the calculation of the spin up using a spin up tolerance
        for tolerance in self._toleranceList:
            assert tolerance in self._spinupToleranceDic
            if self._annId in self._spinupToleranceDic[tolerance]:
                simulationPath.append((True, tolerance, os.path.join('Parameter_{:0>3d}'.format(parameterId), 'SpinupTolerance', 'Tolerance_{:.1e}'.format(tolerance))))

            if self._annId in self._spinupToleranceDicNoMassAdjustment[tolerance]:
                simulationPath.append((True, tolerance, os.path.join('Parameter_{:0>3d}'.format(parameterId), 'SpinupTolerance', 'NoMassAdjustment', 'Tolerance_{:.1e}'.format(tolerance))))

        return simulationPath


    def _checkSimulationFiles(self, simulationPath, parameterId, massAdjustment, tolerance, cpunum=64):
        """
        Check the complettness of the simulation files and the logfile
        @author: Markus Pfeil
        """
        assert simulationPath is not None
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance >= 0
        assert type(cpunum) is int and cpunum > 0

        trajectoryYear = 50 if (tolerance is not None) else 10
        lastYear = 10000 if (tolerance is not None) else 1000
        outputFile = os.path.join(simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        if tolerance is not None and os.path.exists(outputFile) and os.path.isfile(outputFile):
            self._getModelParameter(parameterId)
            metos3d = Metos3d.Metos3d(self._model, self._timestep, self._modelParameter, os.path.join(self._path, simulationPath))
            lastYear = metos3d.lastSpinupYear()

        check = True

        #Check the logfile
        logfileName = Evaluation_Constants.PATTERN_LOGFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, tolerance if tolerance is not None else 0, cpunum)
        logfilePath = os.path.join(self._pathPrediction, 'Logfile', logfileName)
        logfilePathSimulation = os.path.join(simulationPath, logfileName)
        check = check and (os.path.exists(logfilePath) and os.path.isfile(logfilePath) or os.path.exists(logfilePathSimulation) and os.path.isfile(logfilePathSimulation))
        if not check:
            logging.warning('Logfile does not exist: {}\n\n'.format(logfilePath))

        if os.path.exists(simulationPath) and os.path.isdir(simulationPath):
            #Check the job output
            joboutputFile = os.path.join(simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
            check = check and os.path.exists(joboutputFile) and os.path.isfile(joboutputFile)
            if not check:
                logging.warning('Joboutput does not exist: {}\n\n'.format(joboutputFile))

            for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                #Check prediction
                predictionFile = os.path.join(simulationPath, Metos3d_Constants.PATTERN_TRACER_INPUT.format(trac))
                check = check and os.path.exists(predictionFile) and os.path.isfile(predictionFile)
                if not check:
                    logging.warning('Prediction does not exist: {}\n\n'.format(predictionFile))

                for end in ['', '.info']:
                    #Check the tracer
                    tracerPath = os.path.join(simulationPath, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(trac), end))
                    check = check and os.path.exists(tracerPath) and os.path.isfile(tracerPath)
                    if not check:
                        logging.warning('Tracer does not exist: {}\n\n'.format(tracerPath))

                    #Check the tracer of the one step
                    for year in range(trajectoryYear, lastYear, trajectoryYear):
                        tracerfilename = os.path.join(simulationPath, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, trac), end))
                        check = check and os.path.exists(tracerfilename) and os.path.isfile(tracerfilename)
                        if not check:
                            logging.warning('TracerOnestep does not exist: {}\n\n'.format(tracerfilename))
        else:
            check = False
            logging.warning('SimulationPath does not exist: {}'.format(simulationPath))

        #Copy the logfile into the simulation path
        if check and not (os.path.exists(logfilePathSimulation) and os.path.isfile(logfilePathSimulation)):
            shutil.copy2(logfilePath, os.path.join(simulationPath, logfileName))

        return check


    def backup(self, movetar=False):
        """
        Create the backup for the simulation data using the given artificial neural network.
        @author: Markus Pfeil
        """
        assert type(movetar) is bool

        act_path = os.getcwd()
        os.chdir(self._path)

        tarfilename = os.path.join(self._pathTarfile, Evaluation_Constants.PATTERN_BACKUP_FILENAME.format(self._annId, Evaluation_Constants.COMPRESSION))
        assert not os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'w:{}'.format(Evaluation_Constants.COMPRESSION), compresslevel=Evaluation_Constants.COMPRESSLEVEL)

        #Add the ann to the data backup
        for annFile in self._annFilenameList:
            if os.path.exists(annFile) and os.path.isfile(annFile):
                annFilename = os.path.basename(annFile)
                shutil.copy2(annFile, os.path.join(self._path, annFilename))
                try:
                    tar.add(annFilename)
                except tarfile.TarError:
                    logging.warning('Can not add the ann file {} to archiv\n'.format(annFilename))
            else:
                logging.warning('Ann file {} does not exist.\n'.format(annFilename))

        for parameterId in range(0, ANN_Constants.PARAMETERID_MAX_TEST+1):
            for (massAdjustment, tolerance, simPath) in self._generateSimPathPostfix(parameterId):
                if os.path.exists(simPath) and os.path.isdir(simPath) and self._checkSimulationFiles(simPath, parameterId, massAdjustment, tolerance):
                    trajectoryYear = 50 if (tolerance is not None) else 10
                    lastYear = 10001 if (tolerance is not None) else 1001
                    outputFile = os.path.join(simPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                    if tolerance is not None and os.path.exists(outputFile) and os.path.isfile(outputFile):
                        self._getModelParameter(parameterId)
                        metos3d = Metos3d.Metos3d(self._model, self._timestep, self._modelParameter, os.path.join(self._path, simPath))
                        lastYear = metos3d.lastSpinupYear()

                    #Add logfile to the data backup
                    logfileName = Evaluation_Constants.PATTERN_LOGFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, tolerance if tolerance is not None else 0, 64)
                    logfilePath = os.path.join(simPath, logfileName)
                    if os.path.exists(logfilePath) and os.path.isfile(logfilePath):
                        try:
                            tar.add(logfilePath)
                        except tarfile.TarError:
                            logging.warning('Can not add the log file {} to archiv\n'.format(logfilePath))
                    else:
                        logging.warning('Log file {} does not exist.\n'.format(logfilePath))

                    #Add the job output to the data backup
                    joboutputFile = os.path.join(simPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                    if os.path.exists(joboutputFile) and os.path.isfile(joboutputFile):
                        try:
                            tar.add(joboutputFile)
                        except tarfile.TarError:
                            logging.warning('Can not add the job output file {} to archiv\n'.format(joboutputFile))
                    else:
                        logging.warning('Job output file {} does not exist.\n'.format(joboutputFile))

                    for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                        #Add the prediction to the data backup
                        predictionFile = os.path.join(simPath, Metos3d_Constants.PATTERN_TRACER_INPUT.format(trac))
                        if os.path.exists(predictionFile) and os.path.isfile(predictionFile):
                            try:
                                tar.add(predictionFile)
                            except tarfile.TarError:
                                logging.warning('Can not add the predicition file {} to archiv\n'.format(predictionFile))
                        else:
                            logging.warning('Predicition file {} does not exist.\n'.format(predictionFile))

                        for end in ['', '.info']:
                            #Add the tracer
                            tracerFilename = os.path.join(simPath, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(trac), end))
                            if os.path.exists(tracerFilename) and os.path.isfile(tracerFilename):
                                try:
                                    tar.add(tracerFilename)
                                except tarfile.TarError:
                                    logging.warning('Can not add tracer file {} to archiv\n'.format(tracerFilename))
                            else:
                                logging.warning('Tracer file {} does not exist.\n'.format(tracerFilename))

                            #Add the tracer of the one step
                            for year in range(trajectoryYear, lastYear, trajectoryYear):
                                tracerFilename = os.path.join(simPath, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, trac), end))
                                if os.path.exists(tracerFilename) and os.path.isfile(tracerFilename):
                                    try:
                                        tar.add(tracerFilename)
                                    except tarfile.TarError:
                                        logging.warning('Can not add tracer file {} to archiv\n'.format(tracerFilename))
                                else:
                                    logging.warning('Tracer file {} does not exist.\n'.format(tracerFilename))
                else:
                    logging.warning('Path: {} (exists: {}, isDirectory: {}), CheckFiles: {}'.format(simPath, os.path.exists(simPath), os.path.isdir(simPath), self._checkSimulationFiles(simPath, parameterId, massAdjustment, tolerance)))

        tar.close()
        #Move tarfile to TAPE_CACHE
        if movetar:
            shutil.move(tarfilename, os.path.join(Evaluation_Constants.PATH_BACKUP, Evaluation_Constants.PATTERN_BACKUP_FILENAME.format(self._annId, Evaluation_Constants.COMPRESSION)))
        os.chdir(act_path)


    def restore(self, parameterIdList=range(0, ANN_Constants.PARAMETERID_MAX_TEST+1), movetar=False, restoreAnn=True, restoreLogfile=True, restoreJoboutput=True, restorePrediction=True, restoreTracerOnestep=True):
        """
        Restore the simulation data from the backup.
        @author: Markus Pfeil
        """
        assert type(movetar) is bool
        assert type(parameterIdList) is list
        assert type(restoreAnn) is bool
        assert type(restoreLogfile) is bool
        assert type(restoreJoboutput) is bool
        assert type(restorePrediction) is bool
        assert type(restoreTracerOnestep) is bool

        act_path = os.getcwd()
        os.chdir(self._path)

        tarfilename = os.path.join(self._pathTarfile, Evaluation_Constants.PATTERN_BACKUP_FILENAME.format(self._annId, Evaluation_Constants.COMPRESSION))

        #Copy backup file
        if not os.path.exists(tarfilename) and movetar:
            shutil.move(os.path.join(Evaluation_Constants.PATH_BACKUP, Evaluation_Constants.PATTERN_BACKUP_FILENAME.format(self._annId, Evaluation_Constants.COMPRESSION)), tarfilename)

        assert os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'r:{}'.format(Evaluation_Constants.COMPRESSION), compresslevel=Evaluation_Constants.COMPRESSLEVEL)

        os.makedirs(self._path, exist_ok=True)

        #Restore the ann
        if restoreAnn:
            for annFile in self._annFilenameList:
                annFilename = os.path.basename(annFile)
                try:
                    tar.extract(annFilename, path=self._path)
                except (tarfile.TarError, KeyError):
                    logging.warning('There is not the ann file {} in the archiv\n'.format(annFilename))

        for parameterId in parameterIdList:
            for (massAdjustment, tolerance, simPath) in self._generateSimPathPostfix(parameterId):
                os.makedirs(os.path.join(self._path, simPath), exist_ok=True)
                trajectoryYear = 50 if (tolerance is not None) else 10
                lastYear = 10001 if (tolerance is not None) else 1001
                outputFile = os.path.join(simPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                if tolerance is not None and os.path.exists(outputFile) and os.path.isfile(outputFile):
                    self._getModelParameter(parameterId)
                    metos3d = Metos3d.Metos3d(self._model, self._timestep, self._modelParameter, os.path.join(self._path, simPath))
                    lastYear = metos3d.lastSpinupYear()

                #Restore the logfile
                if restoreLogfile:
                    logfileName = Evaluation_Constants.PATTERN_LOGFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, tolerance if tolerance is not None else 0, 64)
                    logfilePath = os.path.join(simPath, logfileName)
                    try:
                        tar.extract(logfilePath, path=self._path)
                    except (tarfile.TarError, KeyError):
                        logging.warning('The log file {} is not in the archiv\n'.format(logfilePath))

                #Restore the job output
                if restoreJoboutput:
                    joboutputFile = os.path.join(simPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                    try:
                        tar.extract(joboutputFile, path=self._path)
                    except (tarfile.TarError, KeyError):
                        logging.warning('The job output {} is not in the archiv\n'.format(joboutputFile))

                for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    #Restore the prediction to the data backup
                    if restorePrediction:
                        predictionFile = os.path.join(simPath, Metos3d_Constants.PATTERN_TRACER_INPUT.format(trac))
                        try:
                            tar.extract(predictionFile, path=self._path)
                        except (tarfile.TarError, KeyError):
                            logging.warning('The prediction file {} is not in the archiv\n'.format(predictionFile))

                    for end in ['', '.info']:
                        #Restore the tracer
                        if restoreTracer:
                            os.makedirs(os.path.join(self._path, simPath, 'Tracer'), exist_ok=True)
                            tracerFilename = os.path.join(simPath, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(trac), end))
                            try:
                                tar.extract(tracerFilename, path=self._path)
                            except (tarfile.TarError, KeyError):
                                logging.warning('The tracer file {} is not in the archiv\n'.format(tracerFilename))

                        #Restore the tracer of the one steps
                        if restoreTracerOnestep:
                            for year in range(trajectoryYear, lastYear, trajectoryYear):
                                tracerFilename = os.path.join(simPath, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, trac), end))
                                try:
                                    tar.extract(tracerFilename, path=self._path)
                                except (tarfile.TarError, KeyError):
                                    logging.warning('The tracer file {} is not in the archiv\n'.format(tracerFilename))

        tar.close()
        os.chdir(act_path)

        #Remove tarfile
        if movetar:
            os.remove(tarfilename)


    def remove(self, movetar=False):
        """
        Remove the simulation data.
        @author: Markus Pfeil
        """
        assert type(movetar) is bool

        tarfilename = os.path.join(self._pathTarfile, Evaluation_Constants.PATTERN_BACKUP_FILENAME.format(self._annId, Evaluation_Constants.COMPRESSION))

        #Copy backup file
        if not os.path.exists(tarfilename) and movetar:
            shutil.move(os.path.join(Evaluation_Constants.PATH_BACKUP, Evaluation_Constants.PATTERN_BACKUP_FILENAME.format(self._annId, Evaluation_Constants.COMPRESSION)), tarfilename)

        assert os.path.exists(tarfilename)
        tar = tarfile.open(tarfilename, 'r:{}'.format(Evaluation_Constants.COMPRESSION), compresslevel=Evaluation_Constants.COMPRESSLEVEL)
       
        #Remove the files of the ANN
        for annFile in self._annFilenameList:
            annFilename = os.path.basename(annFile)
            if os.path.exists(os.path.join(self._path, annFilename)) and os.path.isfile(os.path.join(self._path, annFilename)):
                try:
                    info = tar.getmember(annFilename)
                except KeyError:
                    logging.warning('There is not the file {} of the ANN in the archiv'.format(annFilename))
                else:
                    if info.isfile() and info.size > 0 and os.path.exists(os.path.join(self._path, annFilename)) and os.path.isfile(os.path.join(self._path, annFilename)):
                        os.remove(os.path.join(self._path, annFilename))

        for parameterId in range(0, ANN_Constants.PARAMETERID_MAX_TEST+1):
            for (massAdjustment, tolerance, simPath) in self._generateSimPathPostfix(parameterId):
                if os.path.exists(os.path.join(self._path, simPath)) and os.path.isdir(os.path.join(self._path, simPath)):
                    trajectoryYear = 50 if (tolerance is not None) else 10
                    lastYear = 10001 if (tolerance is not None) else 1001
                    outputFile = os.path.join(simPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                    if tolerance is not None and os.path.exists(os.path.join(self._path, outputFile)) and os.path.isfile(os.path.join(self._path, outputFile)):
                        self._getModelParameter(parameterId)
                        metos3d = Metos3d.Metos3d(self._model, self._timestep, self._modelParameter, os.path.join(self._path, simPath))
                        lastYear = metos3d.lastSpinupYear()

                    #Remove the logfile
                    logfileName = Evaluation_Constants.PATTERN_LOGFILE.format(self._annType, self._model, self._annId, parameterId, massAdjustment, tolerance if tolerance is not None else 0, 64)
                    logfilePath = os.path.join(simPath, logfileName)
                    if os.path.exists(os.path.join(self._path, logfilePath)) and os.path.isfile(os.path.join(self._path, logfilePath)):
                        try:
                            info = tar.getmember(logfilePath)
                        except KeyError:
                            logging.warning('There is not the logfile {} in the archiv\n'.format(logfilePath))
                        else:
                            if info.isfile() and info.size > 0:
                                os.remove(os.path.join(self._path, logfilePath))
                            if info.isfile() and info.size > 0 and os.path.exists(os.path.join(self._pathPrediction, 'Logfile', logfileName)) and os.path.isfile(os.path.join(self._pathPrediction, 'Logfile', logfileName)):
                                os.remove(os.path.join(self._pathPrediction, 'Logfile', logfileName))

                    #Remove the job output
                    joboutputFile = os.path.join(simPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
                    if os.path.exists(os.path.join(self._path, joboutputFile)) and os.path.isfile(os.path.join(self._path, joboutputFile)):
                        try:
                            info = tar.getmember(joboutputFile)
                        except KeyError:
                            logging.warning('There is not the job output file {} in the archiv\n'.format(joboutputFile))
                        else:
                            if info.isfile() and info.size > 0:
                                os.remove(os.path.join(self._path, joboutputFile))

                    for trac in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                        #Remove the prediction
                        predictionFile = os.path.join(simPath, Metos3d_Constants.PATTERN_TRACER_INPUT.format(trac))
                        if os.path.exists(os.path.join(self._path, predictionFile)) and os.path.isfile(os.path.join(self._path, predictionFile)):
                            try:
                                info = tar.getmember(predictionFile)
                            except KeyError:
                                logging.warning('File {} not in the archiv\n'.format(predictionFile))
                            else:
                                if info.isfile() and info.size > 0:
                                    os.remove(os.path.join(self._path, predictionFile))

                        for end in ['', '.info']:
                            #Remove the tracer
                            tracerFilename = os.path.join(simPath, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(trac), end))
                            if os.path.exists(os.path.join(self._path, tracerFilename)) and os.path.isfile(os.path.join(self._path, tracerFilename)):
                                try:
                                    info = tar.getmember(tracerFilename)
                                except KeyError:
                                    logging.warning('File {} not in the archiv\n'.format(tracerFilename))
                                else:
                                    if info.isfile() and info.size > 0:
                                        os.remove(os.path.join(self._path, tracerFilename))

                            #Remove the tracer of the one step
                            for year in range(trajectoryYear, lastYear, trajectoryYear):
                                tracerFilename = os.path.join(simPath, 'TracerOnestep', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(year, trac), end))
                                if os.path.exists(os.path.join(self._path, tracerFilename)) and os.path.isfile(os.path.join(self._path, tracerFilename)):
                                    try:
                                        info = tar.getmember(tracerFilename)
                                    except KeyError:
                                        logging.warning('File {} not in the archiv\n'.format(tracerFilename))
                                    else:
                                        if info.isfile() and info.size > 0:
                                            os.remove(os.path.join(self._path, tracerFilename))
                    #Remove the directories
                    try:
                        for directory in ['Optionfile', 'Tracer', 'TracerOnestep']:
                            directoryPath = os.path.join(self._path, simPath, directory)
                            if os.path.exists(directoryPath) and os.path.isdir(directoryPath) and not os.listdir(directoryPath):
                                os.rmdir(directoryPath)

                        directoryPath = simPath
                        while (directoryPath != '' and os.path.exists(os.path.join(self._path, directoryPath)) and os.path.isdir(os.path.join(self._path, directoryPath))):
                            if not os.listdir(os.path.join(self._path, directoryPath)):
                                os.rmdir(os.path.join(self._path, directoryPath))
                                directoryPath = os.path.dirname(directoryPath)
                            else:
                                break
                    except OSError as ex:
                        if ex.errno == errno.ENOTEMPTY:
                            logging.warning('Directory {} is not empty'.format(directoryPath))

        #Remove directory
        try:
            if not os.listdir(self._path):
                os.rmdir(self._path)
        except OSError as ex:
            if ex.errno == errno.ENOTEMPTY:
                logging.warning('Directory {} is not empty'.format(directoryPath))

        tar.close()

        #Remove tarfile
        if movetar:
            os.remove(tarfilename)

