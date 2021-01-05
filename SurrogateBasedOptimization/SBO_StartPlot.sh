#!/bin/bash

#PBS -N SBO_Plot
#PBS -j o
#PBS -o Plotoutput.out
#PBS -b 1
#PBS -l cpunum_job=32
#PBS -l elapstim_req=2:00:00
#PBS -l memsz_job=48gb
#PBS -T intmpi
#PBS -q clexpress
#
cd $PBS_O_WORKDIR
#
qstat -l -F ehost ${PBS_JOBID/0:}
#
. /sfs/fs2/work-sh1/sunip350/metos3d/nesh_metos3d_setup_v0.6.9.sh
#
export PYTHONPATH=/sfs/fs5/home-sh/sunip350/Python/bgc-ann/util:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/sbo:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/ann:/sfs/fs5/home-sh/sunip350/Python/bgc-ann/SurrogateBasedOptimization
#
python3 /sfs/fs5/home-sh/sunip350/Python/bgc-ann/SurrogateBasedOptimization/SBO_StartPlot.py
export TMPDIR="/scratch/"`echo $PBS_JOBID | cut -f2 -d\:`
~

~

