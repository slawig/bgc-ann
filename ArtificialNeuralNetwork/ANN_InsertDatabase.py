#!/usr/bin/env python
# -*- coding: utf8 -*

import argparse
import os
import shutil
import re
import time
import traceback
import numpy as np
import sqlite3

from ann.database.access import Ann_Database
import ann.network.constants as ANN_Constants
import ANN_Genetic_Algorithm
import metos3dutil.metos3d.Metos3d
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.petsc.petscfile as petsc
import ann.evaluation.constants as Evaluation_Constants
import metos3dutil.database.constants as DB_Constants

import Data_preparing
import util.petsc.universal

PATH_REFERENCE = '/sfs/fs2/work-sh1/sunip350/metos3d/LatinHypercubeSample/Timesteps'


def main(annId=None, massAdjustmentFlag=False, toleranceFlag=False, trajectoryFlag=False):
    if annId is None:
        annIdList = [222]
        massAdjustment = True
        tolerance = 10**(-4)
    else:
        annIdList = [int(annId)]
        tolerance = 10**(-4) if toleranceFlag else None
        massAdjustment = massAdjustmentFlag

    parameterIdList = range(0, ANN_Constants.PARAMETERID_MAX_TEST+1)

    reference = False
    
    trajectoryYear = 50 if (tolerance is not None or (reference and tolerance is None)) else 10
    years = 10000 if (reference or tolerance is not None) else 1000
    
    for annId in annIdList:
        for parameterId in parameterIdList:
            try:
                dbinsert = DatabaseInsert(annId, parameterId, years=years, trajectoryYear=trajectoryYear, massAdjustment=massAdjustment, tolerance=tolerance, reference=reference)          
                print('Insert results into the database for annId {:0>5d} and parameterId {:0>3d} - Jobfile: {} Database: {}/{}/{}/{}/{}'.format(annId, parameterId, dbinsert.existsJoboutputFile(), dbinsert.checkSpinupTotalityDatabase(), dbinsert.checkMassTotalityDatabase(), dbinsert.checkNormTotalityDatabase(), dbinsert.checkDeviationTotalityDatabase(), dbinsert.checkTrajectoryNormTotalityDatabase()))
                
                #Insert spin up norm values
                if dbinsert.existsJoboutputFile() and not dbinsert.checkSpinupTotalityDatabase():
                    dbinsert.insertSpinup(overwrite=True)
                    
                #Insert mass values
                if dbinsert.existsJoboutputFile() and not dbinsert.checkMassTotalityDatabase():
                    dbinsert.calculateMass(overwrite=True)
                
                #Insert norm values
                if dbinsert.existsJoboutputFile() and (not dbinsert.checkNormTotalityDatabase() or not dbinsert.checkDeviationTotalityDatabase()):
                    dbinsert.calculateNorm(overwrite=True)

                #Insert trajectory norm values
                if trajectoryFlag and dbinsert.existsJoboutputFile() and not dbinsert.checkTrajectoryNormTotalityDatabase():
                    dbinsert.calculateTrajectoryNorm(overwrite=True)

            except AssertionError as err:
                print('Assertion error for annId {:0>5d} and parameterId {:0>3d}\n{}'.format(annId, parameterId, err))
                traceback.print_exc()
            except sqlite3.DatabaseError as err:
                print('Database error for annId {:0>5d} and parameterId {:0>3d}\n{}'.format(annId, parameterId, err))
                traceback.print_exc()
            finally:
                try:
                    dbinsert.closeDbConnection()
                except UnboundLocalError as ule:
                    print('No DbInsert instance: {}', ule)

        #Insert ann training progress
        try:
            dbinsert = DatabaseInsert(annId, 0, years=years, trajectoryYear=trajectoryYear, massAdjustment=massAdjustment, tolerance=tolerance, reference=reference)
            if dbinsert.checkAnnTraining():
                dbinsert.insertTraining(overwrite=True)
        except AssertionError as err:
            print('Assertion error for annId {:0>5d} and parameterId {:0>3d}\n{}'.format(annId, parameterId, err))
            traceback.print_exc()
        except sqlite3.DatabaseError as err:
            print('Database error for annId {:0>5d} and parameterId {:0>3d}\n{}'.format(annId, parameterId, err))
            traceback.print_exc()
        finally:
            try:
                dbinsert.closeDbConnection()
            except UnboundLocalError as ule:
                print('No DbInsert instance: {}', ule)



class DatabaseInsert():
    """
    Insert the simulation data of the artificial neural network into the database for the given paramterId.
    @author: Markus Pfeil
    """

    def __init__(self, annId, parameterId, years=1000, trajectoryYear=10, cpunum=64, massAdjustment=False, tolerance=None, reference=False):
        """
        Constructor of the DatabaseInsert class
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(years) is int and years > 0
        assert type(trajectoryYear) is int and trajectoryYear > 0
        assert type(cpunum) is int and cpunum > 0
        assert type(massAdjustment) is bool
        assert tolerance is None or type(tolerance) is float and tolerance > 0
        assert type(reference) is bool
        
        #Database
        self._database = Ann_Database()
        
        #Artificial neural network
        self._annId = annId
        self._annType, self._annNumber, model = self._getAnnConfig()
        
        self._massAdjustment = massAdjustment
        self._tolerance = tolerance if (tolerance is not None) else 0.0 
        self._spinupTolerance = True if (tolerance is not None) else False
        self._reference = reference
        
        #Model
        self._model = model
        self._parameterId = parameterId
        self._years = years
        self._trajectoryYear = trajectoryYear
        
        if self._reference and self._spinupTolerance:
            self._path = os.path.join(ANN_Constants.PATH, 'SpinupTolerance')
        else:
            self._path = os.path.join(ANN_Constants.PATH, 'Prediction')
        self._simulationPath = self._setSimulationPath()
        self._metos3dPath = Metos3d_Constants.METOS3D_PATH
        self._cpunum = cpunum
        
        self._simulationId = self._setSimulationId()


    def closeDbConnection(self):
        """
        Close database connection
        @author: Markus Pfeil
        """
        self._database.close_connection()
        
        
    def _getAnnConfig(self):
        """
        Get the type of the artificial neural network and the model for the id of the ann
        @author: Markus Pfeil
        """
        annType, annNumber, model, conserveMass = self._database.get_annTypeNumberModelMass(self._annId)
        return (annType, annNumber, model)


    def _setSimulationPath(self):
        """
        Create the simulation path
        @author: Markus Pfeil
        """
        if self._reference and self._spinupTolerance:
            simulationPath = os.path.join(self._path, self._model, 'Tolerance_{:.1e}'.format(self._tolerance), 'Parameter_{:0>3d}'.format(self._parameterId))
        elif self._reference and not self._spinupTolerance:
            simulationPath = os.path.join(PATH_REFERENCE, self._model, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(1))
        else:
            simulationPath = os.path.join(self._path, self._model, 'AnnId_{:0>5d}'.format(self._annId), 'Parameter_{:0>3d}'.format(self._parameterId))
            if self._spinupTolerance and self._massAdjustment:
                simulationPath = os.path.join(simulationPath, 'SpinupTolerance', 'Tolerance_{:.1e}'.format(self._tolerance))
            elif self._spinupTolerance:
                simulationPath = os.path.join(simulationPath, 'SpinupTolerance', 'NoMassAdjustment', 'Tolerance_{:.1e}'.format(self._tolerance))
            elif self._massAdjustment:
                simulationPath = os.path.join(simulationPath, 'MassAdjustment')
        return simulationPath


    def _setSimulationId(self):
        """
        Set the simulationId
        @author: Markus Pfeil
        """
        tolerance = self._tolerance if self._spinupTolerance else None
        if self._reference:
            concentrationId = self._database.get_concentrationId_constantValues(self._model, Metos3d_Constants.INITIAL_CONCENTRATION[self._model])
        else:
            concentrationId = self._database.get_concentrationId_annValues(self._annId)
        simulationId = self._database.get_simulationId(self._model, self._parameterId, concentrationId, self._massAdjustment, tolerance=tolerance, cpus=self._cpunum)
        return simulationId


    def existsJoboutputFile(self):
        """
        Test, if the output file exists
        @author: Markus Pfeil
        """
        if self._reference and self._spinupTolerance:
            filename = os.path.join(self._path, 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT_SPINUP_REFERENCE.format(self._model, self._parameterId, self._tolerance, self._cpunum))
        elif self._reference and not self._spinupTolerance:
            return True
        else:
            filename = os.path.join(self._path, 'Joboutput', Evaluation_Constants.PATTERN_JOBOUTPUT.format(self._annType, self._model, self._annId, self._parameterId, self._massAdjustment, self._tolerance, self._cpunum))
        return os.path.exists(filename) and os.path.isfile(filename)


    def checkAnnTraining(self):
        """
        Check, if the database contains all values of the training for each epoch
        @author: Markus Pfeil
        """
        return self._database.check_AnnTraining(self._annId)


    def checkSpinupTotalityDatabase(self):
        """
        Check, if the database contains all values of the spin up norm
        @author: Markus Pfeil
        """
        expectedCount = self._years
        
        if self._spinupTolerance:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._tolerance)
            if lastYear is not None:
                expectedCount = lastYear
        
        return self._database.check_spinup(self._simulationId, expectedCount)


    def checkMassTotalityDatabase(self):
        """
        Check, if the database contains all values of the mass
        @author: Markus Pfeil
        """
        expectedCount = 0
        if not self._reference:
            expectedCount = expectedCount + 1   #Entry of the prediction
        
        years = self._years
        if self._spinupTolerance:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._tolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = expectedCount + years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear
        
        return self._database.check_mass(self._simulationId, expectedCount)


    def checkNormTotalityDatabase(self):
        """
        Check, if the database contains all values of the tracer norm for one calculation
        @author: Markus Pfeil
        """
        expectedCount = 0
        if not self._reference:
            expectedCount = expectedCount + 1   #Entry of the prediction
        
        years = self._years
        if self._spinupTolerance:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._tolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = expectedCount + years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear
        
        checkNorm = True
        
        for norm in DB_Constants.NORM:
            checkNorm = checkNorm and self._database.check_tracer_norm(self._simulationId, expectedCount, norm=norm)
            
            #Norm of the differences
            concentrationIdReference = self._database.get_concentrationId_constantValues(self._model, Metos3d_Constants.INITIAL_CONCENTRATION[self._model])
            simulationIdReference = self._database.get_simulationId(self._model, self._parameterId, concentrationIdReference, False)
            checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, simulationIdReference, expectedCount, norm=norm)
            
            if not self._reference:
                checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, self._simulationId, 1, norm=norm)
        
        return checkNorm


    def checkDeviationTotalityDatabase(self):
        """
        Check, if the database contains all values of the tracer deviation for one calculation
        @author: Markus Pfeil
        """
        expectedCount = 0
        if not self._reference:
            expectedCount = expectedCount + 1   #Entry of the prediction
        
        years = self._years
        if self._spinupTolerance:
            lastYear = self._database.get_spinup_year_for_tolerance(self._simulationId, tolerance=self._tolerance)
            if lastYear is not None:
                years = lastYear
        expectedCount = expectedCount + years//self._trajectoryYear   #Entry for the spin up simulation
        expectedCount = expectedCount + (1 if (years%self._trajectoryYear != 0) else 0) #Entry of the lastYear
        
        checkDeviation = True
        
        checkDeviation = checkDeviation and self._database.check_tracer_deviation(self._simulationId, expectedCount)

        #Norm of the differences
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._model, Metos3d_Constants.INITIAL_CONCENTRATION[self._model])
        simulationIdReference = self._database.get_simulationId(self._model, self._parameterId, concentrationIdReference, False)
        checkDeviation = checkDeviation and self._database.check_difference_tracer_deviation(self._simulationId, simulationIdReference, expectedCount)

        if not self._reference:
            checkDeviation = checkDeviation and self._database.check_difference_tracer_deviation(self._simulationId, self._simulationId, 1)
        
        return checkDeviation


    def checkTrajectoryNormTotalityDatabase(self):
        """
        Check, if the database contains all values of the tracer trajectory norm for one calculation
        @author: Markus Pfeil
        """
        expectedCount = 1
        if not self._reference:
            expectedCount = 2
        
        checkNorm = True
        for norm in DB_Constants.NORM:
            checkNorm = checkNorm and self._database.check_tracer_norm(self._simulationId, expectedCount, norm=norm, trajectory='Trajectory')
            
            if not self._reference:
                #Norm of the differences
                concentrationIdReference = self._database.get_concentrationId_constantValues(self._model, Metos3d_Constants.INITIAL_CONCENTRATION[self._model])
                simulationIdReference = self._database.get_simulationId(self._model, self._parameterId, concentrationIdReference, False)
                checkNorm = checkNorm and self._database.check_difference_tracer_norm(self._simulationId, simulationIdReference, expectedCount, norm=norm, trajectory='Trajectory')
            
        return checkNorm


    def insertTraining(self, overwrite=False):
        """
        Insert the losses for the training of the artificial neural network.
        @author: Markus Pfeil
        """
        assert type(overwrite) is bool

        filenameTraining = ''
        if self._annType == 'fcn':
            filenameTraining = os.path.join(ANN_Constants.PATH, 'Scripts', 'Logfile', 'Joboutput.CreateNeuralNetwork_FCN_{}.log'.format(self._annNumber))
        elif self._annType == 'set':
            filenameTraining = os.path.join(ANN_Constants.PATH_SET, ANN_Constants.FCN_SET.format(self._annNumber), ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(self._annNumber))
        elif self._annType == 'setgen':
            (gen, uid, ann_path) = ANN_Genetic_Algorithm.read_genome_file(self._annNumber)
            filenameTraining = os.path.join(ann_path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(uid))

        if not (os.path.exists(filenameTraining) and os.path.isfile(filenameTraining)):
            logging.error('File {} does not exists.'.format(filenameTraining))
        assert os.path.exists(filenameTraining) and os.path.isfile(filenameTraining)
        print('Filename: {}'.format(filenameTraining)) 
        #Parse trainings monitor
        with open(filenameTraining) as f:
            epoch = None
            for line in f.readlines():
                try:
                    match = re.search(r'^\s*(\d+):\s+(\d+.\d+e[+-]\d+),\s+(\d+.\d+e[+-]\d+),\s+(\d+.\d+e[+-]\d+),\s+(\d+.\d+e[+-]\d+),\s+(\d+.\d+e[+-]\d+),\s+(\d+.\d+e[+-]\d+)', line)
                    matchEpoch = re.search(r'Epoch (\d+)/(\d+)', line)
                    matchTraining = re.search(r'loss: (\d+.\d+) - mean_squared_error: (\d+.\d+) - mean_absolute_error: (\d+.\d+) - val_loss: (\d+.\d+) - val_mean_squared_error: (\d+.\d+) - val_mean_absolute_error: (\d+.\d+)', line)
                    if match:
                        [epoch, loss, valLoss, meanSquaredError, valMeanSquaredError, meanAbsoluteError, valMeanAbsoluteError] = match.groups()
                        self._database.insert_training(self._annId, int(epoch), float(loss), float(valLoss), float(meanSquaredError), float(valMeanSquaredError), float(meanAbsoluteError), float(valMeanAbsoluteError), overwrite=overwrite)
                    elif matchEpoch:
                        [epoch, maxEpoch] = matchEpoch.groups()
                    elif matchTraining:
                        [loss, meanSquaredError, meanAbsoluteError, valLoss, valMeanSquaredError, valMeanAbsoluteError] = matchTraining.groups()
                        assert epoch is not None
                        self._database.insert_training(self._annId, int(epoch), float(loss), float(valLoss), float(meanSquaredError), float(valMeanSquaredError), float(meanAbsoluteError), float(valMeanAbsoluteError), overwrite=overwrite)
                        epoch = None 
                except (sqlite3.IntegrityError, ValueError):
                    logging.error('Inadmissable values for annId {:0>4d} and epoch {:0>4}\n'.format(self._annId, epoch))


    def insertSpinup(self, overwrite=False):
        """
        Insert the spin up norm values of the job_output into the database.
        @author: Markus Pfeil
        """
        filenameJoboutput = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        if not(os.path.exists(filenameJoboutput) and os.path.isfile(filenameJoboutput)):
            print('Filename: {}'.format(filenameJoboutput))
        assert os.path.exists(filenameJoboutput) and os.path.isfile(filenameJoboutput)
        spinup_norm_array = Data_preparing.read_spinup_norm_values(filenameJoboutput)
        spinup_norm_array_shape = np.shape(spinup_norm_array)
        try:
            for i in range(spinup_norm_array_shape[0]):
                year = spinup_norm_array[i,0]
                tolerance = spinup_norm_array[i,1]
                if spinup_norm_array_shape[1] == 3:
                    spinupNorm = spinup_norm_array[i,2]
                else:
                    spinupNorm = None

                if year == 0 and tolerance == 0 and spinupNorm is not None and spinupNorm == 0:
                    raise ValueError()
                self._database.insert_spinup(self._simulationId, year, tolerance, spinupNorm, overwrite=overwrite)
        except (sqlite3.IntegrityError, ValueError):
            logging.error('Inadmissable values for simulationId {:0>4d} and year {:0>4}\n'.format(self._simulationId, year))


    def calculateMass(self, overwrite=False):
        """
        Calculate the mass of the prediction using the ANN
        @author: Markus Pfeil
        """
        vol = self._readBoxVolumes(path=os.path.join(self._metos3dPath, 'data/data/TMM/2.8/Geometry/volumes.petsc'))
        vol_vec = np.empty(shape=(len(vol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            vol_vec[:,i] = vol
        overallMass = sum(Metos3d_Constants.INITIAL_CONCENTRATION[self._model]) * np.sum(vol_vec)
        
        pathMetos3dTracer = os.path.join(self._simulationPath, 'Tracer')
        pathPredictionTracer = self._simulationPath
        pathMetos3dTracerOneStep = os.path.join(self._simulationPath, 'TracerOnestep')
        assert os.path.exists(pathMetos3dTracer) and os.path.isdir(pathMetos3dTracer)
        assert os.path.exists(pathPredictionTracer) and os.path.isdir(pathPredictionTracer)
        assert os.path.exists(pathMetos3dTracerOneStep) and os.path.isdir(pathMetos3dTracerOneStep)
        
        if not self._reference:
            tracerPrediction = self._getTracerOutput(pathPredictionTracer, Metos3d_Constants.PATTERN_TRACER_INPUT, year=None)
            massPrediction = np.sum(tracerPrediction * vol_vec)
            self._database.insert_mass(self._simulationId, 0, massPrediction/overallMass, massPrediction-overallMass, overwrite=overwrite)
        
        lastYear = self._years
        outputFile = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        if self._spinupTolerance and os.path.exists(outputFile) and os.path.isfile(outputFile):
            lastYear = Data_preparing.last_year(outputFile)
        
        for year in range(self._trajectoryYear, lastYear, self._trajectoryYear):
            tracerMetos3d = self._getTracerOutput(pathMetos3dTracerOneStep, Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR, year=year)
            massMetos3d = np.sum(tracerMetos3d * vol_vec)
            self._database.insert_mass(self._simulationId, year, massMetos3d/overallMass, massMetos3d-overallMass, overwrite=overwrite)
        
        tracerMetos3d = self._getTracerOutput(pathMetos3dTracer, Metos3d_Constants.PATTERN_TRACER_OUTPUT, year=None)
        #Mass of the tracer
        massMetos3d = np.sum(tracerMetos3d * vol_vec)
        self._database.insert_mass(self._simulationId, lastYear, massMetos3d/overallMass, massMetos3d-overallMass, overwrite=overwrite)


    def calculateNorm(self, overwrite=False):
        """
        Calculate the norm values for every tracer output.
        @author: Markus Pfeil
        """
        #Read box volumes
        normvol = self._readBoxVolumes()
        vol = self._readBoxVolumes(path=os.path.join(self._metos3dPath, 'data/data/TMM/2.8/Geometry/volumes.petsc'))
        euclidean = np.ones(shape=(len(normvol)))
        
        normvol_vec = np.empty(shape=(len(normvol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
        vol_vec = np.empty(shape=(len(vol), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
        euclidean_vec = np.empty(shape=(len(euclidean), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            normvol_vec[:,i] = normvol
            vol_vec[:,i] = vol
            euclidean_vec[:,i] = euclidean
        normWeight = {'2': euclidean_vec, 'Boxweighted': normvol_vec, 'BoxweightedVol': vol_vec}
        
        #Tracer's directories
        pathReferenceTracer = os.path.join(ANN_Constants.PATH, 'Tracer', self._model, 'Parameter_{:0>3d}'.format(self._parameterId))
        pathPredictionTracer = self._simulationPath
        pathMetos3dTracer = os.path.join(self._simulationPath, 'Tracer')
        pathMetos3dTracerOneStep = os.path.join(self._simulationPath, 'TracerOnestep')
        
        assert os.path.exists(pathReferenceTracer) and os.path.isdir(pathReferenceTracer)
        assert os.path.exists(pathPredictionTracer) and os.path.isdir(pathPredictionTracer)
        assert os.path.exists(pathMetos3dTracer) and os.path.isdir(pathMetos3dTracer)
        assert os.path.exists(pathMetos3dTracerOneStep) and os.path.isdir(pathMetos3dTracerOneStep)

        #Read tracer
        tracerReference = self._getTracerOutput(pathReferenceTracer, Metos3d_Constants.PATTERN_TRACER_OUTPUT, year=None)
        yearReference = 10000
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._model, Metos3d_Constants.INITIAL_CONCENTRATION[self._model])
        simulationIdReference = self._database.get_simulationId(self._model, self._parameterId, concentrationIdReference, False)

        #Insert the tracer norm of the metos3d calculation
        if not self._reference:
            #Insert the norm of the prediction using the ann
            tracerPrediction = self._getTracerOutput(pathPredictionTracer, Metos3d_Constants.PATTERN_TRACER_INPUT, year=None)
            for norm in DB_Constants.NORM:
                #Insert the norm values
                self._calculateTracerNorm(tracerPrediction, 0, norm, normWeight[norm], overwrite=overwrite)
                self._calculateTracerDifferenceNorm(0, simulationIdReference, yearReference, tracerPrediction, tracerReference, norm, normWeight[norm], overwrite=overwrite)
       
            self._calculateTracerDeviation(tracerPrediction, 0, overwrite=overwrite)
            self._calculateTracerDifferenceDeviation(0, simulationIdReference, yearReference, tracerPrediction, tracerReference, overwrite=overwrite)
     
        lastYear = self._years
        outputFile = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        if self._spinupTolerance and os.path.exists(outputFile) and os.path.isfile(outputFile):
            lastYear = Data_preparing.last_year(outputFile)
            
        for year in range(self._trajectoryYear, lastYear, self._trajectoryYear):
            #Read tracer
            tracerMetos3dYear = self._getTracerOutput(pathMetos3dTracerOneStep, Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR, year=year)
            
            for norm in DB_Constants.NORM:
                #Insert the norm values
                self._calculateTracerNorm(tracerMetos3dYear, year, norm, normWeight[norm], overwrite=overwrite)
                self._calculateTracerDifferenceNorm(year, simulationIdReference, yearReference, tracerMetos3dYear, tracerReference, norm, normWeight[norm], overwrite=overwrite)
                
            self._calculateTracerDeviation(tracerMetos3dYear, year, overwrite=overwrite)
            self._calculateTracerDifferenceDeviation(year, simulationIdReference, yearReference, tracerMetos3dYear, tracerReference, overwrite=overwrite)

        tracerMetos3d = self._getTracerOutput(pathMetos3dTracer, Metos3d_Constants.PATTERN_TRACER_OUTPUT, year=None)
        for norm in DB_Constants.NORM:
            #Insert the norm values
            self._calculateTracerNorm(tracerMetos3d, lastYear, norm, normWeight[norm], overwrite=overwrite)
            self._calculateTracerDifferenceNorm(lastYear, simulationIdReference, yearReference, tracerMetos3d, tracerReference, norm, normWeight[norm], overwrite=overwrite)
            if not self._reference:
                self._calculateTracerDifferenceNorm(lastYear, self._simulationId, 0, tracerMetos3d, tracerPrediction, norm, normWeight[norm], overwrite=overwrite)
           
        self._calculateTracerDeviation(tracerMetos3d, lastYear, overwrite=overwrite)
        self._calculateTracerDifferenceDeviation(lastYear, simulationIdReference, yearReference, tracerMetos3d, tracerReference, overwrite=overwrite)
        if not self._reference:
            self._calculateTracerDifferenceDeviation(lastYear, self._simulationId, 0, tracerMetos3d, tracerPrediction, overwrite=overwrite)
       

    def _calculateTracerNorm(self, tracer, year, norm, normWeight, overwrite=False):
        """
        Calculate the norm of a tracer
        @author: Markus Pfeil
        """
        assert tracer is not None
        assert year >= 0
        assert norm in DB_Constants.NORM
        assert normWeight is not None
        assert overwrite or not overwrite
        
        tracerNorm = np.sqrt(np.sum((tracer)**2 * normWeight))
        tracerSingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            tracerSingleValue[t] = np.sqrt(np.sum((tracer[:,t])**2 * normWeight[:,t]))
        
        self._database.insert_tracer_norm_tuple(self._simulationId, year, tracerNorm, tracerSingleValue[0], DOP=tracerSingleValue[1], P=tracerSingleValue[2], Z=tracerSingleValue[3], D=tracerSingleValue[4], norm=norm, overwrite=overwrite)


    def _calculateTracerDifferenceNorm(self, year, simulationId, yearB, tracerA, tracerB, norm, normWeight, overwrite=False):
        """
        Calculate the norm of the difference of two tracers
        @author: Markus Pfeil
        """
        assert simulationId >= 0
        assert year >= 0
        assert yearB >= 0
        assert tracerA is not None
        assert tracerB is not None
        assert norm in DB_Constants.NORM
        assert normWeight is not None

        tracerDifferenceNorm = np.sqrt(np.sum((tracerA - tracerB)**2 * normWeight))
        tracerSingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            tracerSingleValue[t] = np.sqrt(np.sum((tracerA[:,t] - tracerB[:,t])**2 * normWeight[:,t]))
        self._database.insert_difference_tracer_norm_tuple(self._simulationId, simulationId, year, yearB, tracerDifferenceNorm, tracerSingleValue[0], DOP=tracerSingleValue[1], P=tracerSingleValue[2], Z=tracerSingleValue[3], D=tracerSingleValue[4], norm=norm, overwrite=overwrite)


    def _calculateTracerDeviation(self, tracer, year, overwrite=False):
        """
        Calculate the deviation for the given tracer
        @author: Markus Pfeil
        """
        assert year is None or year >= 0
        assert tracer is not None

        #Calculation of the mean value for all tracers
        mean = np.mean(tracer, axis=0)
        assert len(mean) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        #Calculation of the variance value for all tracers
        var = np.var(tracer, axis=0)
        assert len(var) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        #Calculation of the minimal values for all tracers
        minimum = np.nanmin(tracer, axis=0)
        assert len(minimum) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        #Calculation of the maximal values for all tracers
        maximal = np.nanmax(tracer, axis=0)
        assert len(maximal) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])

        meanValue = [None, None, None, None, None]
        varValue = [None, None, None, None, None]
        minimumValue = [None, None, None, None, None]
        maximumValue = [None, None, None, None, None]
        negativeCountValue = [None, None, None, None, None]
        negativeSumValue = [None, None, None, None, None]
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            meanValue[i] = mean[i]
            varValue[i] = var[i]
            minimumValue[i] = minimum[i]
            maximumValue[i] = maximal[i]
            #Calculation of the count of boxes with negative concentrations
            negativeCountValue[i] = np.count_nonzero(tracer[tracer[:,i]<0, i])
            #Calculation of the sum of all negative concentrations
            negativeSumValue[i] = np.sum(tracer[tracer[:,i]<0, i])
    
        self._database.insert_deviation_tracer_tuple(self._simulationId, year, meanValue[0], varValue[0], minimumValue[0], maximumValue[0], negativeCountValue[0], negativeSumValue[0], DOP_mean=meanValue[1], DOP_var=varValue[1], DOP_min=minimumValue[1], DOP_max=maximumValue[1], DOP_negative_count=negativeCountValue[1], DOP_negative_sum=negativeSumValue[1], P_mean=meanValue[2], P_var=varValue[2], P_min=minimumValue[2], P_max=maximumValue[2], P_negative_count=negativeCountValue[2], P_negative_sum=negativeSumValue[2], Z_mean=meanValue[3], Z_var=varValue[3], Z_min=minimumValue[3], Z_max=maximumValue[3], Z_negative_count=negativeCountValue[3], Z_negative_sum=negativeSumValue[3], D_mean=meanValue[4], D_var=varValue[4], D_min=minimumValue[4], D_max=maximumValue[4], D_negative_count=negativeCountValue[4], D_negative_sum=negativeSumValue[4], overwrite=overwrite)


    def _calculateTracerDifferenceDeviation(self, year, simulationId, yearB, tracerA, tracerB, overwrite=False):
        """
        Calculate the deviation for the given tracer
        @author: Markus Pfeil
        """
        assert year >= 0
        assert yearB >= 0
        assert simulationId >= 0
        assert tracerA is not None
        assert tracerB is not None

        tracer = np.fabs(tracerA - tracerB)

        #Calculation of the mean value for all tracers
        mean = np.mean(tracer, axis=0)
        assert len(mean) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        #Calculation of the variance value for all tracers
        var = np.var(tracer, axis=0)
        assert len(var) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        #Calculation of the minimal values for all tracers
        minimum = np.nanmin(tracer, axis=0)
        assert len(minimum) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        #Calculation of the maximal values for all tracers
        maximal = np.nanmax(tracer, axis=0)
        assert len(maximal) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])

        meanValue = [None, None, None, None, None]
        varValue = [None, None, None, None, None]
        minimumValue = [None, None, None, None, None]
        maximumValue = [None, None, None, None, None]
        negativeCountValue = [None, None, None, None, None]
        negativeSumValue = [None, None, None, None, None]
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            meanValue[i] = mean[i]
            varValue[i] = var[i]
            minimumValue[i] = minimum[i]
            maximumValue[i] = maximal[i]
            #Calculation of the count of boxes with negative concentrations
            negativeCountValue[i] = np.count_nonzero(tracer[tracer[:,i]<0, i])
            #Calculation of the sum of all negative concentrations
            negativeSumValue[i] = np.sum(tracer[tracer[:,i]<0, i])
    
        self._database.insert_difference_tracer_deviation_tuple(self._simulationId, simulationId, year, yearB, meanValue[0], varValue[0], minimumValue[0], maximumValue[0], negativeCountValue[0], negativeSumValue[0], DOP_mean=meanValue[1], DOP_var=varValue[1], DOP_min=minimumValue[1], DOP_max=maximumValue[1], DOP_negative_count=negativeCountValue[1], DOP_negative_sum=negativeSumValue[1], P_mean=meanValue[2], P_var=varValue[2], P_min=minimumValue[2], P_max=maximumValue[2], P_negative_count=negativeCountValue[2], P_negative_sum=negativeSumValue[2], Z_mean=meanValue[3], Z_var=varValue[3], Z_min=minimumValue[3], Z_max=maximumValue[3], Z_negative_count=negativeCountValue[3], Z_negative_sum=negativeSumValue[3], D_mean=meanValue[4], D_var=varValue[4], D_min=minimumValue[4], D_max=maximumValue[4], D_negative_count=negativeCountValue[4], D_negative_sum=negativeSumValue[4], overwrite=overwrite)


    def _readBoxVolumes(self, path='/sfs/fs5/home-sh/sunip350/.metos3d/data/data/TMM/2.8/Geometry/normalizedVolumes.petsc', tracer_length=52749):
        """
        Read volumes of the boxes
        @author: Markus Pfeil
        """
        assert tracer_length > 0
        
        f = open(path, 'rb')
        #Jump over the header
        np.fromfile(f, dtype='>i4', count=2)
        normvol = np.fromfile(f, dtype='>f8', count=tracer_length)
        return normvol


    def _getTracerOutput(self, path, tracerPattern, year=None, tracer_length=52749):
        """
        Read concentration values of all tracer for a given year
        @author: Markus Pfeil
        """
        assert year is None or year >= 0
        assert tracer_length > 0
        
        tracer_array = np.empty(shape=(tracer_length, len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            if year is None:
                filename = tracerPattern.format(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model][i])
            else:
                filename = tracerPattern.format(year, Metos3d_Constants.METOS3D_MODEL_TRACER[self._model][i])
            tracerfile = os.path.join(path, filename)
            assert os.path.exists(tracerfile) and os.path.isfile(tracerfile)
            tracer = util.petsc.universal.load_petsc_vec_to_numpy_array(tracerfile)
            assert len(tracer) == tracer_length
            tracer_array[:,i] = tracer
        return tracer_array


    def calculateTrajectoryNorm(self, timestep=1, overwrite=False):
        """
        Calculation of all trajectory norm relevant values
        @author: Markus Pfeil
        """
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(overwrite) is bool
        
        #Read box volumes
        normvol = self._readBoxVolumes()
        vol = self._readBoxVolumes(path=os.path.join(self._metos3dPath, 'data/data/TMM/2.8/Geometry/volumes.petsc'))
        euclidean = np.ones(shape=(len(normvol)))

        normvol_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        vol_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        euclidean_vec = np.empty(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
        for t in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)):
            for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
                normvol_vec[t,i,:] = normvol
                vol_vec[t,i,:] = vol
                euclidean_vec[t,i,:] = euclidean
        normWeight = {'2': euclidean_vec, 'Boxweighted': normvol_vec, 'BoxweightedVol': vol_vec}

        #Trajectory directories
        trajectoryPath = os.path.join(self._simulationPath, 'Trajectory')
        referenceTrajectoryPath = os.path.join(ANN_Constants.PATH, 'Tracer', self._model, 'Parameter_{:0>3d}'.format(self._parameterId), 'Trajectory')
        os.makedirs(trajectoryPath, exist_ok=True)
        os.makedirs(referenceTrajectoryPath, exist_ok=True)
        assert os.path.exists(trajectoryPath) and os.path.isdir(trajectoryPath)
        assert os.path.exists(referenceTrajectoryPath) and os.path.isdir(referenceTrajectoryPath)
        
        #Read reference trajectory
        yearReference = 10000
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._model, Metos3d_Constants.INITIAL_CONCENTRATION[self._model])
        simulationIdReference = self._database.get_simulationId(self._model, self._parameterId, concentrationIdReference, False)
        trajectoryReference = self._calculateTrajectory(referenceTrajectoryPath, year=yearReference, timestep=timestep)
       
        for norm in DB_Constants.NORM:
            if not self._database.check_tracer_norm(simulationIdReference, 1, norm=norm, trajectory='Trajectory'):
                self._calculateTrajectoryTracerNorm(trajectoryReference, yearReference, norm, normWeight[norm], timestep=timestep, simulationId=simulationIdReference, overwrite=overwrite)

        lastYear = self._years
        outputFile = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        if self._spinupTolerance and os.path.exists(outputFile) and os.path.isfile(outputFile):
            lastYear = Data_preparing.last_year(outputFile)

        for (year, last) in [(0, False), (lastYear, True)]:
            #Read trajectory
            trajectoryPathYear = os.path.join(trajectoryPath, 'Year_{:0>5}'.format(year), '{:d}dt'.format(timestep))
            os.makedirs(trajectoryPathYear, exist_ok=True)
            trajectory = self._calculateTrajectory(trajectoryPathYear, year=year, lastYear=last, timestep=timestep)

            for norm in DB_Constants.NORM:
                self._calculateTrajectoryTracerNorm(trajectory, year, norm, normWeight[norm], timestep=timestep, overwrite=overwrite)
                self._calculateTrajectoryDifferenceTracerNorm(year, simulationIdReference, yearReference, trajectory, trajectoryReference, norm, normWeight[norm], timestep=timestep, overwrite=overwrite)
                
        #Remove the directory for the trajectory
        shutil.rmtree(trajectoryPath, ignore_errors=True)
        
        
    def _existsTrajectory(self, trajectoryPath, timestep=1):
        """
        Check, if the trajectory exists
        @author: Markus Pfeil
        """
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        
        trajectoryExists = os.path.exists(trajectoryPath) and os.path.isdir(trajectoryPath)
        if trajectoryExists:
            for index in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    trajectoryFilename = Metos3d_Constants.PATTERN_TRACER_TRAJECTORY.format(0, index, tracer)
                    trajectoryExists = trajectoryExists and os.path.exists(os.path.join(trajectoryPath, trajectoryFilename)) and os.path.isfile(os.path.join(trajectoryPath, trajectoryFilename))
        return trajectoryExists


    def _calculateTrajectoryTracerNorm(self, trajectory, year, norm, normWeight, timestep=1, simulationId=None, overwrite=False):
        """
        Calculate trajectory norm for a given trajectory
        @author: Markus Pfeil
        """
        assert trajectory is not None
        assert type(year) is int and year >= 0
        assert norm in DB_Constants.NORM
        assert normWeight is not None
        assert type(overwrite) is bool
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert simulationId is None or type(simulationId) is int and 0 <= simulationId

        dt = 1.0 / int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)
        trajectoryNorm = np.sqrt(np.sum(trajectory**2 * normWeight) * dt)
        trajectorySingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            trajectorySingleValue[t] = np.sqrt(np.sum((trajectory[:,t,:])**2 * normWeight[:,t,:]) * dt)
        
        simId = self._simulationId if simulationId is None else simulationId
        self._database.insert_tracer_norm_tuple(simId, year, trajectoryNorm, trajectorySingleValue[0], DOP=trajectorySingleValue[1], P=trajectorySingleValue[2], Z=trajectorySingleValue[3], D=trajectorySingleValue[4], norm=norm, trajectory='Trajectory', overwrite=overwrite)


    def _calculateTrajectoryDifferenceTracerNorm(self, year, simulationId, yearB, trajectoryA, trajectoryB, norm, normWeight, timestep=1, overwrite=False):
        """
        Calculate trajectory norm between the difference of two trajectories
        @author: Markus Pfeil
        """
        assert simulationId >= 0
        assert type(year) is int and year >= 0
        assert type(yearB) is int and yearB >= 0
        assert trajectoryA is not None
        assert trajectoryB is not None
        assert norm in DB_Constants.NORM
        assert normWeight is not None
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        dt = 1.0 / int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / timestep)
        trajectoryDifferenceNorm = np.sqrt(np.sum((trajectoryA - trajectoryB)**2 * normWeight) * dt)
        trajectorySingleValue = [None, None, None, None, None]
        for t in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])):
            trajectorySingleValue[t] = np.sqrt(np.sum((trajectoryA[:,t,:] - trajectoryB[:,t,:])**2 * normWeight[:,t,:]) * dt)

        self._database.insert_difference_tracer_norm_tuple(self._simulationId, simulationId, year, yearB, trajectoryDifferenceNorm, trajectorySingleValue[0], DOP=trajectorySingleValue[1], P=trajectorySingleValue[2], Z=trajectorySingleValue[3], D=trajectorySingleValue[4], norm=norm, trajectory='Trajectory', overwrite=overwrite)


    def _calculateTrajectory(self, metos3dSimulationPath, year=0, lastYear=False, timestep=1, modelYears=0, queue='clexpress', cores=1):
        """
        Calculate the tracer concentration for the target model parameter
        @author: Markus Pfeil
        """
        assert os.path.exists(metos3dSimulationPath) and os.path.isdir(metos3dSimulationPath)
        assert year in range(0, 10001)
        assert type(lastYear) is bool
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        
        #Run metos3d
        tracer_path = os.path.join(metos3dSimulationPath, 'Tracer')
        os.makedirs(tracer_path, exist_ok=True)
        
        modelParameter = list(self._database.get_parameter(self._parameterId, self._model))
        model = Metos3d.Metos3d(self._model, timestep, modelParameter, metos3dSimulationPath, modelYears = modelYears, queue = queue, cores = cores)
        model.setCalculateOnlyTrajectory()
        
        if not self._existsTrajectory(tracer_path, timestep=timestep):
            #Copy the input tracer for the trajectory
            if year == 0:
                inputTracer = os.path.join(self._simulationPath, 'N_input.petsc')
            elif lastYear:
                inputTracer = os.path.join(self._simulationPath, 'Tracer', 'N_output.petsc')
            elif year == 10000:
                inputTracer = os.path.join(os.path.dirname(metos3dSimulationPath), 'N_output.petsc')
            else:
                inputTracer = self._getTracerOutput(os.path.join(self._simulationPath, 'TracerOnestep'), Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR, year=year)
            shutil.copy(inputTracer, os.path.join(tracer_path, 'N_input.petsc'))
            
            model.run()

        #Read tracer concentration
        trajectory = model.readTracer()
        
        return trajectory



if __name__ == '__main__':
    #Command line parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("-annId", nargs='?', const=None, default=None, help="Insert the data for the given annId")
    parser.add_argument("-massAdjustment", "--massAdjustment", action="store_true")
    parser.add_argument("-tolerance", "--tolerance", action="store_true")
    parser.add_argument("-trajectory", "--trajectory", action="store_true")   
 
    args = parser.parse_args() 
    main(annId=args.annId, massAdjustmentFlag=args.massAdjustment, toleranceFlag=args.tolerance, trajectoryFlag=args.trajectory)
