# bgc-ann

Marine ecosystem models are important to identify the processes that affects for example the global carbon cycle. Computation of an annually periodic solution (i.e., a steady annual cycle) for these models requires a high computational effort. To reduce this effort, we approximated an exemplary marine ecosystem model by different artificial neural networks (ANNs). We used a fully connected network, then applied the sparse evolutionary training (SET) procedure, and finally applied a genetic algorithm (GA) to optimize both the network topology.



## Installation

To clone this project with **git** run:
>git clone https://github.com/slawig/bgc-ann.git



## Usage

The project consists of two python packages (util and ann) as well as python script to start the training and evaluation of the approximation using ANNs (Directory ArtificialNeuralNetwork)


### Python package util

This package contains function the run [Metos3D](https://metos3d.github.io/) (Marine Ecosystem Toolkit for Optimization and Simulation in 3-D) on the NEC HPC-Linux-Cluster of the CAU Kiel.


### Python package ann

This package contains four subpackages:
- network:
  Contains basic functions for using neural networks and the SET algorithm. 
- database:
  Consists of functions to store the results of the approximation in a database and read them out again.
- evaluation:
  Summary of functions to calculate the approximations using the prediction of a ANN.
- geneticAlgorithm:
  Contains functions to train a ANN using a genetic algorithm.


### Python scripts

There are three groups of scripts:

1. Creation and training of ANNs:
   The script `ANN_CreateNeuralNetwork.py` can be used to start the creation and training of new ANNs. For this purpose, the configuration of the ANN is read from the configuration file `ANN_Config_FCN.py` (for a fully connected network) or `ANN_Config_SET.py` (using the SET algorithm).
2. Evaluation of the approximation using a ANN:
   - The script `ANN_Evaluation.py` starts the evaluation of the approximation of a steady annual cycle by a ANN. For this purpose, the script `ANN_EvaluationJob.py` is called for each parameter vector of the test data to compute different approximations using the prediction of the ANN.
   - The script `ANN_InsertDatabase.py` writes the evaluation results into a database.
3. Visualization of the results:
   - The script `ANN_Plotfunction.py` provides different functions to visualize the results.
   - The script `Plots_ANN_Paper.py` generates from the data provided on [Zenodo](https://doi.org/10.5281/zenodo.4058319) the figures that are shown in the draft of the paper with the title "Approximation of a marine ecosystem model by artificial neural networks designed using a genetic algorithm". A description of how to use this script is included in the Wiki.



## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

