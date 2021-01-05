#!/usr/bin/env python
# -*- coding: utf8 -*

import os


PATTERN_LOGFILE = '{}_Logfile.{}.AnnId_{:0>3d}.ParameterId_{:0>3d}.MassAdjustment_{}.SpinupTolerance_{:.1e}.cpunum_{:0>3d}.log'
PATTERN_LOGFILE_SPINUP_REFERENCE = 'Logfile.{}.ParameterId_{:0>3d}.SpinupTolerance_{:.1e}.cpunum_{:0>3d}.log'
PATTERN_JOBFILE = '{}_Jobfile.{}.AnnId_{:0>3d}.ParameterId_{:0>3d}.MassAdjustment_{}.SpinupTolerance_{:.1e}.cpunum_{:0>3d}.txt'
PATTERN_JOBFILE_SPINUP_REFERENCE = 'Jobfile.{}.ParameterId_{:0>3d}.SpinupTolerance_{:.1e}.cpunum_{:0>3d}.txt'
PATTERN_JOBOUTPUT = '{}_Joboutput.{}.AnnId_{:0>3d}.ParameterId_{:0>3d}.MassAdjustment_{}.SpinupTolerance_{:.1e}.cpunum_{:0>3d}.out'
PATTERN_JOBOUTPUT_SPINUP_REFERENCE = 'Joboutput.{}.ParameterId_{:0>3d}.SpinupTolerance_{:.1e}.cpunum_{:0>3d}.out'

#Backup
PATH_BACKUP = os.path.join('/nfs', 'tape_cache', 'sunip350', 'Daten', 'metos3d', 'ArtificialNeuralNetwork', 'Simulation')
PATTERN_BACKUP_FILENAME = 'ANN_SimulationDataBackup_AnnId_{:0>5d}.tar.{}'
PATTERN_BACKUP_LOGFILE = 'ANN_Backup_OptimizationId_{:0>4d}_Backup_{}_Remove_{}_Restore_{}.log'
COMPRESSION = 'bz2'
COMPRESSLEVEL = 9

