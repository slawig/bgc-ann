#!/bin/bash

#SBATCH --job-name=SBO_Plot
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=Plotoutput.out
#SBATCH --partition=cluster
#SBATCH --qos=test

export OMP_NUM_THREADS=1

source /gxfs_work1/cau/sunip350/metos3d/nesh_metos3d_setup.sh
export PYTHONPATH=$HOME/Python/bgc-ann/util:$HOME/Python/bgc-ann/ann:$HOME/Python/bgc-ann/sbo:$HOME/Python/bgc-ann/ArtificialNeuralNetwork:$HOME/Python/bgc-ann/SurrogateBasedOptimization

python3 /gxfs_home/cau/sunip350/Python/bgc-ann/SurrogateBasedOptimization/SBO_StartPlot.py 339

jobinfo

