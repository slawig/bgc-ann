#!/bin/bash

#SBATCH --job-name=ANN_SET
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=48:00:00
#SBATCH --output=/gxfs_work1/cau/sunip350/metos3d/ArtificialNeuralNetwork/Scripts/Joboutput.CreateNeuralNetwork_SET_16.log
#SBATCH --partition=cluster
#SBATCH --qos=normal

export OMP_NUM_THREADS=1

source /gxfs_work1/cau/sunip350/metos3d/nesh_metos3d_setup.sh
export PYTHONPATH=/gxfs_home/cau/sunip350/Python/bgc-ann/util:/gxfs_home/cau/sunip350/Python/bgc-ann/ann:/gxfs_home/cau/sunip350/Python/bgc-ann/ArtificialNeuralNetwork

python3 /gxfs_home/cau/sunip350/Python/bgc-ann/ArtificialNeuralNetwork/ANN_CreateNeuralNetwork.py "set" 16

jobinfo

