#!/bin/bash

#PBS -N ANN_SET
#PBS -j o
#PBS -o /sfs/fs2/work-sh1/sunip350/metos3d/ArtificialNeuralNetwork/Scripts/Joboutput.CreateNeuralNetwork_SET_16.log
#PBS -b 1
#PBS -l cpunum_job=32
#PBS -l elapstim_req=48:00:00
#PBS -l memsz_job=20gb
#PBS -T intmpi
#PBS -q clmedium
cd $PBS_O_WORKDIR
#
qstat -l -F ehost ${PBS_JOBID/0:}
#
. /sfs/fs2/work-sh1/sunip350/metos3d/nesh_metos3d_setup_v0.6.9.sh
#
export PYTHONPATH=/sfs/fs5/home-sh/sunip350/Python/bgc-ann/util:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/ann:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/ArtificialNeuralNetwork
#
python3 /sfs/fs5/home-sh/sunip350/Python/bgc-ann/ArtificialNeuralNetwork/ANN_CreateNeuralNetwork.py "set" 16
export TMPDIR="/scratch/"`echo $PBS_JOBID | cut -f2 -d\:`

