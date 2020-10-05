#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import time
import subprocess
import multiprocessing as mp
import threading
import logging
import numpy as np
import re

import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.petsc.petscfile as petsc
import neshCluster.constants as NeshCluster_Constants


class Metos3d():
    """
    Functions to run metos3D
    @author: Markus Pfeil
    """

    def __init__(self, model, timestep, modelParameter, simulationPath, modelYears = 10000, queue = 'clmedium', cores = 2):
        """
        Initialisation for metos3d
        @author: Markus Pfeil
        """
        assert type(model) is str and model in Metos3d_Constants.METOS3D_MODELS
        assert type(timestep) is int and timestep in Metos3d_Constants.METOS3D_TIMESTEPS
        assert type(modelParameter) is list and len(modelParameter) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[model]
        assert type(modelYears) is int and 0 <= modelYears
        assert os.path.exists(simulationPath) and os.path.isdir(simulationPath)
        assert type(queue) is str and queue in NeshCluster_Constants.QUEUE
        assert type(cores) is int and 0 < cores
        
        #Logging
        self.queue = mp.Queue()
        self.logger = logging.getLogger(__name__)
        self.lp = threading.Thread(target=self.logger_thread)
        
        #Model
        self._model = model
        self._timestep = timestep
        self._modelParameter = modelParameter
        self._modelYears = modelYears
        self._tolerance = 0.0
        self._modelPath = os.path.join(Metos3d_Constants.METOS3D_MODEL_PATH, self._model, Metos3d_Constants.PATTERN_MODEL_FILENAME.format(self._model))
        self._simulationPath = simulationPath
        self._trajectory = False
        self._trajectoryStep = None
        self._trajectoryYear = 50
        self._onlyTrajectory = False

        self._oneStep = False
        self._oneStepYear = 50
        self._tracerInputDir = None

        

        self._options = {} 

        self._queue = queue
        self._cores = cores
        
        logging.info('***Metos3d***\nModel: {}\nTimestep id: {:d}\nModel parameter: {}\nSpin-up over {:d} years\n'.format(self._model, self._timestep, self._modelParameter, self._modelYears))


    def logger_thread(self):
        """
        Logging for multiprocessing
        @author: Markus Pfeil
        """
        while True:
            record = self.queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)


    def setTolerance(self, tolerance):
        """
        Set the tolerance for the calculation of the spin up using metos3d
        @author: Markus Pfeil
        """
        assert type(tolerance) is float and 0 <= tolerance
        
        self._tolerance = tolerance


    def setTrajectoryParameter(self, trajectoryYear=50, trajectoryStep=None):
        """
        Set the parameter of the monitor to save tracer concentration during the spin up.
        @author: Markus Pfeil
        """
        assert type(trajectoryYear) is int and 0 < trajectoryYear
        assert trajectoryStep is None or type(trajectoryStep) is int and 0 < trajectoryStep < Metos3d_Constants.METOS3D_STEPS_PER_YEAR

        self._trajectory = True
        self._trajectoryYear = trajectoryYear
        self._trajectoryStep = trajectoryStep


    def setCalculateTrajectory(self):
        """
        Calculate the trajectory for the last model year
        @author: Markus Pfeil
        """
        self._trajectory = True
        self._modelYears = self._modelYears + 1


    def setCalculateOnlyTrajectory(self):
        """
        Calculate the trajectory for the last model year
        @author: Markus Pfeil
        """
        self.setCalculateTrajectory()
        self._onlyTrajectory = True


    def setInputDir(self, inputPath):
        """
        Set the path for the tracer used as initial concentration.
        @author: Markus Pfeil
        """
        assert os.path.exists(inputPath) and os.path.isdir(inputPath)

        self._inputDir = inputPath


    def setOneStep(self, oneStepYear=50):
        """
        Set the parameter using for the simulation with Metos3d of only one step.
        @author: Markus Pfeil
        """
        assert type(oneStepYear) is int and 0 < oneStepYear

        self._oneStep = True
        self._oneStepYear = oneStepYear


    def run(self):
        """
        Start the simulation
        @author: Markus Pfeil
        """
        timeStart = time.time()
        self._startSimulation()
        timeSimulation = time.time()
        self.logger.debug('Time for simulation: {:.6f}s\n\n'.format(timeSimulation - timeStart))


    def readTracer(self):
        """
        Read the calculated tracer concentration
        @author: Markus Pfeil
        """
        if self._trajectory:
            tracerConcentration = np.zeros(shape=(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR/self._timestep), len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN))
            
            for i in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR/self._timestep)):
                j = 0
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    filename = os.path.join(self._simulationPath, 'Tracer', Metos3d_Constants.PATTERN_TRACER_TRAJECTORY.format(self._modelYears - 1, i, tracer))
                    tracerConcentration[i, j, :] = petsc.readPetscFile(filename)
                    j = j + 1
        else:
            tracerConcentration = np.zeros(shape=(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]), Metos3d_Constants.METOS3D_VECTOR_LEN))

            i = 0
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                filename = os.path.join(self._simulationPath, 'Tracer', Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
                tracerConcentration[i, :] = petsc.readPetscFile(filename)
                i = i + 1
        
        return tracerConcentration


    def removeTracer(self, oneStepYear=None):
        """
        Remove the calculated tracer of the trajectory
        @author: Markus Pfeil
        """
        assert oneStepYear is None or type(oneStepYear) is int and 0 < oneStepYear

        if oneStepYear is not None:
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                for end in ['', '.info']:
                    filename = os.path.join(self._simulationPath, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_TRAJECTORY.format(oneStepYear - 1, int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR/self._timestep - 1), tracer), end))
                    if os.path.exists(filename) and os.path.isfile(filename):
                        os.remove(filename)
        elif self._trajectory:
            for i in range(int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR/self._timestep)):
                for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]:
                    for end in ['', '.info']:
                        filename = os.path.join(self._simulationPath, 'Tracer', '{}{}'.format(Metos3d_Constants.PATTERN_TRACER_TRAJECTORY.format(self._modelYears - 1, i, tracer), end))
                        if os.path.exists(filename) and os.path.isfile(filename):
                            os.remove(filename)


    def saveNumpyArrayAsPetscFile(self, filename, vec):
        """
        Save the numpy array as petsc file
        @author: Markus Pfeil
        """
        assert type(vec) is np.ndarray and vec.shape == (Metos3d_Constants.METOS3D_VECTOR_LEN,)
        path = os.path.join(self._simulationPath, 'Tracer')
        os.makedirs(path, exist_ok=True)
        petsc.writePetscFile(os.path.join(path, filename), vec)


    def _startSimulation(self):
        """
        Start the simulation
        @author: Markus Pfeil
        """
        optionfile = self._optionfile()
        x = subprocess.run("mpirun $NQSII_MPIOPTS -np {:d} {} {}\n".format(self._cores * NeshCluster_Constants.CPUNUM[self._queue], self._modelPath, optionfile), shell=True, stdout=subprocess.PIPE, check=True)
        stdout_str = x.stdout.decode(encoding='UTF-8')
        with open(os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME), mode='w') as fid:
            fid.write(stdout_str)
        self.logger.debug('Output of the simulation\n{}'.format(stdout_str))
        os.remove(optionfile)


    def _optionfile(self):
        """
        Create the option file for metos3d using the given parameter
        @author: Markus Pfeil
        """
        optionfile = Metos3d_Constants.PATTERN_OPTIONFILE.format(self._model, self._timestep)
        self._setOptionsOptionfile()
        self._writeOptionfile(os.path.join(self._simulationPath, optionfile))
        return os.path.join(self._simulationPath, optionfile)


    def _setOptionsOptionfile(self):
        """
        Generate dictionary with the options for metos3d
        @author: Markus Pfeil
        """
        self._options = {}
        self._options['/metos3d/debuglevel'] = 1
        self._options['/metos3d/data_dir'] = os.path.join(Metos3d_Constants.METOS3D_PATH, 'data/data/TMM/2.8')

        self._options['/metos3d/tracer_count'] = len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])
        self._options['/metos3d/tracer_name'] = Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]
        
        self._options['/model/initial_concentrations'] = Metos3d_Constants.INITIAL_CONCENTRATION[self._model]
        assert len(self._options['/model/initial_concentrations']) == len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._model])

        if self._tracerInputDir is not None:
            self._options['/metos3d/tracer_input_dir'] = self._tracerInputDir
            options['/metos3d/input_filenames'] = [Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer) for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]]

        self._options['/metos3d/tracer_output_dir'] = os.path.join(self._simulationPath, 'Tracer')
        os.makedirs(self._options['/metos3d/tracer_output_dir'], exist_ok=True)
        self._options['/metos3d/output_filenames'] = [Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer) for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]]

        self._options['/model/parameters'] = self._modelParameter
        self._options['/metos3d/parameters_string'] = ','.join(map(str, self._modelParameter))

        self._options['/model/time_step_multiplier'] = self._timestep
        time_steps_per_year = int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR / self._timestep)
        self._options['/model/time_steps_per_year'] = time_steps_per_year
        self._options['/model/time_step'] = 1 / time_steps_per_year

        self._options['/metos3d/tolerance'] = self._tolerance

        self._options['/metos3d/years'] = self._modelYears

        self._options['/metos3d/write_trajectory'] = self._trajectory
        self._options['/model/Monitor'] = [self._trajectoryYear, time_steps_per_year if self._trajectoryStep is None else self._trajectoryStep]

        self._options['/metos3d/diagnostic_count'] = 0
        
        if self._oneStep:
            self._options['/metos3d/tracer_input_dir'] = os.path.join(self._simulationPath, 'Tracer')
            options['/metos3d/input_filenames'] = [Metos3d_Constants.PATTERN_TRACER_TRAJECTORY.format(int(self._oneStepYear - 1), int(Metos3d_Constants.METOS3D_STEPS_PER_YEAR/self._timestep - 1), tracer) for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]]    
            self._options['/metos3d/tracer_output_dir'] = os.path.join(self._simulationPath, 'TracerOnestep')
            os.makedirs(self._options['/metos3d/tracer_output_dir'], exist_ok=True)
            self._options['/metos3d/output_filenames'] = [Metos3d_Constants.PATTERN_TRACER_OUTPUT_YEAR.format(self._oneStepYear, tracer) for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]]
            self._options['/metos3d/write_trajectory'] = False
            self._options['/metos3d/years'] = 1
            self._options['/model/time_steps_per_year'] = 1
        elif self._onlyTrajectory:
            self._options['/metos3d/tracer_input_dir'] = os.path.join(self._simulationPath, 'Tracer')
            self._options['/metos3d/input_filenames'] = [Metos3d_Constants.PATTERN_TRACER_INPUT.format(tracer) for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._model]]
            self._options['/metos3d/write_trajectory'] = True
            self._options['/metos3d/years'] = 1
            self._options['/model/Monitor'] = [1, 1 if self._trajectoryStep is None else self._trajectoryStep]

    
    def _writeOptionfile(self, optionFilename):
        """
        Create option file with the given option parameters
        @author: Markus Pfeil
        """
        assert not os.path.exists(optionFilename)
        
        f = open(optionFilename, mode='w')

        f.write('# debug \n')
        f.write('-Metos3DDebugLevel                      {:d} \n\n'.format(self._options['/metos3d/debuglevel']))

        f.write('# geometry \n')
        f.write('-Metos3DGeometryType                    Profile \n')
        f.write('-Metos3DProfileInputDirectory           {}/Geometry/ \n'.format(self._options['/metos3d/data_dir']))
        f.write('-Metos3DProfileMaskFile                 landSeaMask.petsc \n')
        f.write('-Metos3DProfileVolumeFile               volumes.petsc \n\n')

        f.write('# bgc tracer \n')
        f.write('-Metos3DTracerCount                     {} \n'.format(self._options['/metos3d/tracer_count']))
        f.write('-Metos3DTracerName                      {} \n'.format(','.join(map(str, self._options['/metos3d/tracer_name']))))
        
        try:
            f.write('-Metos3DTracerInputDirectory            {}/ \n'.format(self._options['/metos3d/tracer_input_dir']))
            f.write('-Metos3DTracerInitFile                  {} \n'.format(','.join(map(str, self._options['/metos3d/input_filenames']))))
        except KeyError:
            f.write('-Metos3DTracerInitValue                 {} \n'.format(','.join(map(str, self._options['/model/initial_concentrations']))))

        f.write('-Metos3DTracerOutputDirectory           {}/ \n'.format(self._options['/metos3d/tracer_output_dir']))
        f.write('-Metos3DTracerOutputFile                {} \n'.format(','.join(map(str, self._options['/metos3d/output_filenames']))))
        f.write('-Metos3DTracerMonitor \n\n')
        
        f.write('# diagnostic variables\n')
        f.write('-Metos3DDiagnosticCount                 {} \n\n'.format(self._options['/metos3d/diagnostic_count']))

        f.write('# bgc parameter \n')
        f.write('-Metos3DParameterCount                  {:d} \n'.format(len(self._options['/model/parameters'])))
        f.write('-Metos3DParameterValue                  {} \n\n'.format(self._options['/metos3d/parameters_string']))

        f.write('# bgc boundary conditions \n')
        f.write('-Metos3DBoundaryConditionCount          2 \n')
        f.write('-Metos3DBoundaryConditionInputDirectory {}/Forcing/BoundaryCondition/ \n'.format(self._options['/metos3d/data_dir']))
        f.write('-Metos3DBoundaryConditionName           Latitude,IceCover \n')
        f.write('-Metos3DLatitudeCount                   1 \n')
        f.write('-Metos3DLatitudeFileFormat              latitude.petsc \n')
        f.write('-Metos3DIceCoverCount                   12 \n')
        f.write('-Metos3DIceCoverFileFormat              fice_$02d.petsc \n\n')

        f.write('# bgc domain conditions \n')
        f.write('-Metos3DDomainConditionCount            2 \n')
        f.write('-Metos3DDomainConditionInputDirectory   {}/Forcing/DomainCondition/ \n'.format(self._options['/metos3d/data_dir']))
        f.write('-Metos3DDomainConditionName             LayerDepth,LayerHeight \n')
        f.write('-Metos3DLayerDepthCount                 1 \n')
        f.write('-Metos3DLayerDepthFileFormat            z.petsc \n\n')
        f.write('-Metos3DLayerHeightCount                1 \n')
        f.write('-Metos3DLayerHeightFileFormat           dz.petsc \n\n')

        f.write('# transport \n')
        f.write('-Metos3DTransportType                   Matrix \n')
        f.write('-Metos3DMatrixInputDirectory            {}/Transport/Matrix5_4/{:d}dt/ \n'.format(self._options['/metos3d/data_dir'], self._options['/model/time_step_multiplier']))
        f.write('-Metos3DMatrixCount                     12 \n')
        f.write('-Metos3DMatrixExplicitFileFormat        Ae_$02d.petsc \n')
        f.write('-Metos3DMatrixImplicitFileFormat        Ai_$02d.petsc \n\n')

        f.write('# time stepping \n')
        if self._oneStep:
            f.write('-Metos3DTimeStepStart                   {:.18f} \n'.format(1.0-self._options['/model/time_step']))
        else:
            f.write('-Metos3DTimeStepStart                   0.0 \n')
        f.write('-Metos3DTimeStepCount                   {:d} \n'.format(self._options['/model/time_steps_per_year']))
        f.write('-Metos3DTimeStep                        {:.18f} \n\n'.format(self._options['/model/time_step']))

        f.write('# solver \n')
        f.write('-Metos3DSolverType                      Spinup \n')
        f.write('-Metos3DSpinupMonitor \n')
        try:
            f.write('-Metos3DSpinupTolerance                 {:f} \n'.format(self._options['/metos3d/tolerance']))
        except KeyError:
            pass
        f.write('-Metos3DSpinupCount                     {:d} \n'.format(self._options['/metos3d/years']))
        
        if self._options['/metos3d/write_trajectory']:
            f.write('-Metos3DSpinupMonitorFileFormatPrefix   sp$0004d-,ts$0004d- \n')
            f.write('-Metos3DSpinupMonitorModuloStep         {},{} \n'.format(*self._options['/model/Monitor']))

        f.close()


    def lastSpinupYear(self):
        """
        Read the last year of the spin up using the output file of the spin up calculation.
        @author: Markus Pfeil
        """
        filename = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        assert os.path.exists(filename) and os.path.isfile(filename)

        last_spinup_line = None
        with open(filename, 'r') as f:
            for line in f.readlines():
                if 'Spinup Function norm' in line:
                    last_spinup_line = line

        if last_spinup_line is not None:
            last_spinup_line = last_spinup_line.strip()
            last_spinup_year_str = last_spinup_line.split()[1]
            last_spinup_year = int(last_spinup_year_str) + 1
        else:
            last_spinup_year = 0

        return last_spinup_year


    def read_spinup_norm_values(self, tolerance_only=False):
        """
        Read the values of the spin up norm from the output file of the spin up calculation.
        @author: Markus Pfeil
        """
        filename = os.path.join(self._simulationPath, Metos3d_Constants.PATTERN_OUTPUT_FILENAME)
        assert os.path.exists(filename) and os.path.isfile(filename)

        years = self.lastSpinupYear()

        if tolerance_only:
            array_shape = (years, 2)
        else:
            array_shape = (years, 3)
        spinup_norm_array = np.zeros(shape = array_shape)

        with open(filename, 'r') as f:
            i = 0
            for line in f:
                if tolerance_only:
                    matches = re.search(r'^\s*\d+.\d+s (\d+) Spinup Function norm (\d+.\d+e[+-]\d+)', line)
                    if matches:
                        [year, tolerance] = matches.groups()
                        spinup_norm_array[i,0] = int(year)
                        spinup_norm_array[i,1] = float(tolerance)
                        i = i + 1
                else:
                    matches = re.search(r'^\s*\d+.\d+s (\d+) Spinup Function norm (\d+.\d+e[+-]\d+) (\d+.\d+e[+-]\d+)', line)
                    if matches:
                        [year, tolerance, spinup_norm] = matches.groups()
                        spinup_norm_array[i,0] = int(year)
                        spinup_norm_array[i,1] = float(tolerance)
                        spinup_norm_array[i,2] = float(spinup_norm)
                        i = i + 1

        return spinup_norm_array


def readBoxVolumes(normvol=False):
    """
    Read volumes of the boxes
    @author: Markus Pfeil
    """
    assert type(normvol) is bool

    path = os.path.join(Metos3d_Constants.METOS3D_PATH, 'data/data/TMM/2.8/Geometry/normalizedVolumes.petsc' if normvol else 'data/data/TMM/2.8/Geometry/volumes.petsc')
    with open(path, 'rb') as f:
        #Jump over the header
        np.fromfile(f, dtype='>i4', count=2)
        normvol = np.fromfile(f, dtype='>f8', count=Metos3d_Constants.METOS3D_VECTOR_LEN)

    return normvol

