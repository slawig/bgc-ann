#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import numpy as np
import logging

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras import backend as K

from sklearn.model_selection import KFold

from ann.network.ANN import ANN as ANN
import ann.network.constants as ANN_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


class FCN(ANN):
    """
    Implementation of a fully connected network using keras
    @author: Markus Pfeil
    """

    def __init__(self, annNumber, metos3dModel='N'):
        """
        Initialization
        @author: Markus Pfeil
        """
        assert type(annNumber) is int and 0 <= annNumber
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        ANN.__init__(self, annNumber, metos3dModel=metos3dModel)

        # Special parameter
        self._path = os.path.join(ANN_Constants.PATH_FCN, self._metos3dModel, ANN_Constants.FCN.format(self._annNumber))
        os.makedirs(self._path, exist_ok=True)
        self._validation_split = None
        self._kfold_splits = None
        self._kfold_shuffle = True
        self._kfold_random_state = None


    def set_validation_spilt(self, validation_split):
        """

        @author: Markus Pfeil
        """
        assert type(validation_split) is float and 0.0 <= validation_split and validation_split <= 1.0

        self._validation_split = validation_split


    def set_kFold(self, kfold_splits, kfold_shuffle=True, kfold_random_state=None):
        """

        @author: Markus Pfeil
        """
        assert type(kfold_splits) is int and 2 >= kfold_splits
        assert type(kfold_shuffle) is bool
        assert kfold_random_state is None or type(kfold_random_state) is int

        self._kfold_splits = kfold_splits
        self._kfold_shuffle = kfold_shuffle
        self._kfold_random_state = kfold_random_state


    def create_model(self, unitsPerLayer=[10, 25, 211], activation=['elu', 'elu', 'elu', 'selu'], dropout=None):
        """
        Create model of the fully connected network
        'unitsPerLayer' includes only the hidden layer. 'activation' includes also an activation for the output layer.
        @author: Markus Pfeil
        """
        assert type(unitsPerLayer) is list
        assert type(activation) is list
        assert len(unitsPerLayer) + 1 == len(activation)  #Last activation function is for the output layer
        assert dropout is None or type(dropout) is float and 0.0 < dropout and dropout < 1.0

        # Append the number of nerons of the output layer
        unitsPerLayer.append(Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel])

        # Build the artificial neural network
        self._model = Sequential()
        self._model.add(Dense(unitsPerLayer[0], input_dim=Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._metos3dModel], name=ANN_Constants.ANN_LAYER_NAME.format(0), activation=activation[0]))
        
        for i in range(1, len(unitsPerLayer)):
            if dropout is not None:
                self._model.add(Dropout(dropout))
            self._model.add(Dense(unitsPerLayer[i], name=ANN_Constants.ANN_LAYER_NAME.format(i), activation=activation[i]))


    def loadAnn(self):
        """
        Load the fully connected network from file
        @autor: Markus Pfeil
        """
        fcnFilename = os.path.join(self._path, ANN_Constants.ANN_FILENAME_FCN.format(self._annNumber))
        assert os.path.exists(fcnFilename) and os.path.isfile(fcnFilename)
        self._model = load_model(fcnFilename)


    def saveAnn(self):
        """
        Save the fully connected network
        @author: Markus Pfeil
        """
        fcnFilename = os.path.join(self._path, ANN_Constants.ANN_FILENAME_FCN.format(self._annNumber))
        assert not os.path.exists(fcnFilename)
        self._model.save(fcnFilename)


    def train(self):
        """
        Train the artificial neural networtk
        @author: Markus Pfeil
        """
        # Read training and test data
        (trainX, testX, trainY, testY) = self._read_data()
        
        # Set the monitoring
        self._losses_per_epoch = {'loss': [], 'val_loss': []}
        for metric in self._metrics:
            self._losses_per_epoch[metric] = []
            self._losses_per_epoch['val_{}'.format(metric)] = []

        if self._kfold_splits is None:
            self._trainHelper(trainX, trainY, testX, testY)
        else:
            kfold = KFold(n_splits=self._kfold_splits, shuffle=self._kfold_shuffle, random_state=self._kfold_random_state)
            for train, test in kfold.split(trainX, trainY):
                self._trainHelper(trainX[train], trainY[train], trainX[test], trainY[test])


    def _trainHelper(self, trainX, trainY, testX, testY, patience=2):
        """

        @author: Markus Pfeil
        """
        sgd = optimizers.SGD(lr=self._learning_rate, momentum=self._momentum, nesterov=self._nesterov)
        self._model.compile(loss=self._loss, optimizer=sgd, metrics=self._metrics)

        if self._validation_split is None:
            historytemp = self._model.fit(trainX, trainY, epochs=self._maxepoches, batch_size=self._batch_size, validation_data=(testX, testY))
        else:
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
            historytemp = self._model.fit(trainX, trainY, epochs=self._maxepoches, batch_size=self._batch_size, validation_split=self._validation_split, vallbacks=[early_stopping]) 

        # Monitoring the training step
        for metric in self._losses_per_epoch:
            self._losses_per_epoch[metric].append(historytemp.history[metric][0])
            self._losses_per_epoch[metric] = np.asarray(self._losses_per_epoch[metric])

