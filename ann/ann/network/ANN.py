#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import os
import numpy as np

import ann.network.constants as ANN_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.latinHypercubeSample.lhs as lhs
import metos3dutil.petsc.petscfile as petsc

class ANN(ABC):
    """
    Abstract basis class of the artificial neural network implementation.
    @author: Markus Pfeil
    """

    def __init__(self, annNumber, metos3dModel='N'):
        """
        Initialization of the parameter for the artificial neural network
        @author: Markus Pfeil
        """
        assert type(annNumber) is int and 0 <= annNumber
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        self._annNumber = annNumber
        self._metos3dModel = metos3dModel

        # Set parameter of the artificial neural network
        self._model = None                  # Artificial neural network

        # Set model parameters
        self._batch_size = 20               # Batch size
        self._maxepoches = 1000             # Number of epoches
        self._learning_rate = 0.01          # SGD learning rate
        self._momentum = 0.0                # SGD momentum
        self._nesterov = False              # Use nesterov momentum
        self._loss = 'mean_squared_error'   # Loss function
        self._alpha = 0.5                   # Weight for the loss function
        self._metrics = ['mean_squared_error', 'mean_absolute_error']  # Metrics to monitor the training

        # Set parameter for the test data 
        self._percentageTestData = 0.2      # Percentage of the whole test data for the validation
        self._dataIndices = np.arange(ANN_Constants.PARAMETERID_MAX_TEST+1, ANN_Constants.PARAMETERID_MAX+1)  # Indices of the data sets used for training and validation

        # Set the path to the dirctory
        self._path = None


    def set_batch_size(self, batch_size):
        """
        Set the batch size for the training.
        @author: Markus Pfeil
        """
        assert type(batch_size) is int and batch_size > 0
        self._batch_size = batch_size


    def set_maxepoches(self, maxepoches):
        """
        Set the maximal number of epoches for the training.
        @author: Markus Pfeil
        """
        assert type(maxepoches) is int and maxepoches > 0
        self._maxepoches = maxepoches


    def set_learning_rate(self, learning_rate):
        """
        Set the SGD learning rate for the training.
        @author: Markus Pfeil
        """
        assert type(learning_rate) is float and learning_rate > 0
        self._learning_rate = learning_rate


    def set_momentum(self, momentum):
        """
        Set the SGD momentum for the training.
        @author: Markus Pfeil
        """
        assert type(momentum) is float and 0 <= momentum and momentum <= 1
        self._momentum = momentum


    def set_nesterov(self, nesterov):
        """
        Set the SGD nesterov for the training.
        @author: Markus Pfeil
        """
        assert type(nesterov) is bool
        self._nesterov = nesterov


    def set_loss(self, loss):
        """
        Set the loss function for the training.
        @author: Markus Pfeil
        """
        assert type(loss) is str
        self._loss = loss


    def set_alpha(self, alpha):
        """
        Set the weight for the loss function for the training.
        @author: Markus Pfeil
        """
        assert type(alpha) is float and 0<= alpha and alpha <= 1.0
        self._alpha = alpha


    def set_metrics(self, metrics):
        """
        Set the metrics for the monitoring during the training
        Use the expanded name because the name is also used to monitor the training.
        @author: Markus Pfeil
        """
        assert type(metrics) is list
        self._metrics = metrics


    def set_percentageTestData(self, percentageTestData):
        """
        Set the percentage of the data sets for the relationship between the training and validation data sets.
        @author: Markus Pfeil
        """
        assert type(percentageTestData) is float and 0.0 <= percentageTestData and percentageTestData <= 1.0
        self._percentageTestData = percentageTestData


    def set_dataIndicesList(self, dataIndices):
        """
        Set the indices for the data sets used for the training and validation of the artificial neural network.
        @author: Markus Pfeil
        """
        assert type(dataIndices) is list
        for i in range(len(dataIndices)):
            assert type(dataIndices[i]) is int and 0<= dataIndices[i] and dataIndices[i] <=  ANN_Constants.PARAMETERID_MAX

        self._dataIndices = np.array(sorted(dataIndices))


    def set_dataIndices(self, indexMin=0, indexMax=ANN_Constants.PARAMETERID_MAX):
        """
        Set the indices for the data sets used for the training and validation of the artificial neural network.
        @author: Markus Pfeil
        """
        assert type(indexMin) is int
        assert 0 <= indexMin
        assert indexMin <= ANN_Constants.PARAMETERID_MAX 
        assert type(indexMin) is int and 0 <= indexMin and indexMin <= ANN_Constants.PARAMETERID_MAX
        assert type(indexMax) is int and indexMin < indexMax and indexMax <= ANN_Constants.PARAMETERID_MAX

        self._dataIndices = np.arange(indexMin, indexMax+1)


    def set_path(self, path):
        """
        Set the path for the neural network.
        @author: Markus Pfeil
        """
        os.makedirs(path, exist_ok=True)
        self._path = path
        assert os.path.exists(self._path) and os.path.isdir(self._path)


    def _read_data(self):
        """
        Read the data sets for training and validation
        The value percentageTestData describes the part of the data used as validation data while the rest of the data is used as trainings data.
        @author: Markus Pfeil
        """
        indices = np.random.permutation(self._dataIndices)
        x_data, y_data = self._loadDataSets(indices)

        # Spilt the data sets in data sets for the training and the validation
        length = int(indices.shape[0] - np.ceil(indices.shape[0] * self._percentageTestData))
        return (x_data[:length], x_data[length:], y_data[:length], y_data[length:])


    def _loadDataSets(self, indices):
        """
        Load the parameter and tracer concentration values of the latin hypercube sample for the given indicies.
        @author: Markus Pfeil
        """
        x = np.zeros(shape=(len(indices), Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._metos3dModel]))
        y = np.zeros(shape=(len(indices), Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel]))

        i = 0
        for parameterId in indices:
            for tracer in Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel]:
                x[i, :] = lhs.readParameterValues(parameterId, self._metos3dModel)
                filename = os.path.join(ANN_Constants.PATH_TRACER, self._metos3dModel, 'Parameter_{:0>3d}'.format(parameterId), Metos3d_Constants.PATTERN_TRACER_OUTPUT.format(tracer))
                y[i, :] = petsc.readPetscFile(filename)
            i = i + 1

        return x, y


    def saveTrainingMonitoring(self):
        """
        Save the monitoring of the training
        @author: Markus Pfeil
        """
        assert os.path.exists(self._path) and os.path.isdir(self._path)
        filename = os.path.join(self._path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(self._annNumber))
        assert not os.path.exists(filename)
        with open(filename, 'w') as f:
            f.write('Epoch: ')
            f.write(', '.join(map(str, [metric for metric in self._losses_per_epoch])))
            f.write('\n\n')
            for epoch in range(len(self._losses_per_epoch['loss'])):
                f.write('{:>5d}: '.format(epoch))
                f.write(', '.join(map(str, ['{:.10e}'.format(self._losses_per_epoch[metric][epoch]) for metric in self._losses_per_epoch])))
                f.write('\n')


    @abstractmethod
    def create_model(self, unitsPerLayer=[10, 25, 211], activation=['elu', 'elu', 'elu', 'selu'], dropout=None):
        """
        Subclasses have to implement the method creating the artificial neural network.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def loadAnn(self):
        """
        Subclasses have to implement the method loading the artificial neural network from the file system.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def saveAnn(self):
        """
        Subclasses have to implement the method saving the artificial neural network to the file system.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def train(self):
        """
        Subclasses have to implement the method training the artificial neural network.
        @author: Markus Pfeil
        """
        pass


    def predict(self, parameter):
        """
        Calculate the prediction for the given parameterId or the given list of parameter values
        The function raises an assertion error if the neural network does not exists.
        @author: Markus Pfeil
        """
        assert type(parameter) is int or type(parameter) is list
        
        if type(parameter) is int:
            assert 0 <= parameter and parameter <= ANN_Constants.PARAMETERID_MAX
            parameterValues = lhs.readParameterValues(parameter, self._metos3dModel)
        else:
            assert len(parameter) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._metos3dModel]
            parameterValues = parameter
            
        tracer_concentrations = self._model.predict(np.array([parameterValues]))
        
        tracer = np.zeros(shape=(Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel], len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            tracer[:, i] = tracer_concentrations[0][i * Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel] : (i+1) * Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel]]
        return tracer

