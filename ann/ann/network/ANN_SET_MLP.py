#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import numpy as np
import logging

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Layer
from keras import optimizers
from keras import backend as K
from keras.utils import Sequence

from ann.network.ANN import ANN as ANN
import ann.network.constants as ANN_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants
import metos3dutil.petsc.petscfile as petsc



class Constraint(object):
    """
    The class was taken from https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks/blob/master/SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py.
    @author: Mocanu, Decebal Constantin
    """

    def __call__(self, w):
        return w


    def get_config(self):
        return {}



class MaskWeights(Constraint):
    """
    The class was taken from https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks/blob/master/SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py.
    @author: Mocanu, Decebal Constantin
    """

    def __init__(self, mask):
        self.mask = np.array(mask)


    def __call__(self, w):
        w = w * K.cast(self.mask, K.floatx())
        return w


    def get_config(self):
        return {'mask': self.mask.astype(np.dtype('i1'))}



class TracerSequence(Sequence):
    """
    Sequence of datasets using biogeochemical tracers
    This class implements the required functions for the Sequence class.
    @author: Markus Pfeil
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        Initialisation of the sequence
        @author: Markus Pfeil
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size


    def __len__(self):
        """
        @author: Markus Pfeil
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))


    def __getitem__(self, idx):
        """
        @author: Markus Pfeil
        """
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x, batch_y



class ConserveMass(Layer):
    """
    The ConserveMass is a layer in order to conserve the mass in the neural network.
    @author: Markus Pfeil
    """

    def __init__(self, overallMass, vol, **kwargs):
        """
        Initialize the conserveMass layer
        @author: Markus Pfeil
        """
        super(ConserveMass, self).__init__(**kwargs)
        self.overallMass = overallMass
        self.vol = vol


    def call(self, inputs):
        """
        @author: Markus Pfeil
        """
        #Set negative concentration values zo zero
        tracer_concentration = K.relu(inputs)

        #Mass of the tracer concentration/prediction using the ANN
        massInputs = K.sum(tracer_concentration * self.vol)

        return (self.overallMass / massInputs) * inputs


    def get_config(self):
        """
        Serialization of the layer
        @author: Markus Pfeil
        """
        config = super(ConserveMass, self).get_config()
        config.update({'overallMass': self.overallMass, 'vol': self.vol})
        return config



class SET_MLP(ANN):
    """
    Sparse evolutionary training (SET)
    The implementation of this algorithm is based on the implementation of Mocanu: https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks/blob/master/SET-MLP-Keras-Weights-Mask/set_mlp_keras_cifar10.py
    @author: Markus Pfeil
    """

    def __init__(self, annNumber, metos3dModel='N', setPath=True):
        """
        Initialization of SET algorithm
        @author: Markus Pfeil
        """
        assert type(annNumber) is int and 0 <= annNumber
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        ANN.__init__(self, annNumber, metos3dModel=metos3dModel)
        
        # Set special parameter for the SET algorithm
        self._path = os.path.join(ANN_Constants.PATH_SET, ANN_Constants.FCN_SET.format(self._annNumber))
        if setPath:
            os.makedirs(self._path, exist_ok=True)
        self._weights = {}              # Weights for every layer
        self._weightsMask = {}          # weights mask for every layer
        self._noParameters = {}         # Number of parameters for every layer
        
        # Set model parameters
        self._epsilon = 20              # Control the sparsity level
        self._zeta = 0.3                # Fraction of the weights removed
        self._SETepoches = 1            # Number of epoches without chancing the network architecture
    
        # Set parameter for a layer conserving mass
        self._conserveMass = False      # Use as last layer a layer to conserve mass
        self._overallMass = None
        self._vol_vec = None
    

    def set_epsilon(self, epsilon):
        """
        Set the control of the sparsity level of the SET algorithm.
        @author: Markus Pfeil
        """
        assert type(epsilon) is float and epsilon > 0
        self._epsilon = epsilon


    def set_zeta(self, zeta):
        """
        Set the fraction of the weights removed of the SET algorithm.
        @author: Markus Pfeil
        """
        assert type(zeta) is float and 0 <= zeta and zeta <= 1
        self._zeta = zeta


    def set_SETepoches(self, SETepoches):
        """
        Set the number of epoches for the training with the same network architecture.
        @author: Markus Pfeil
        """
        assert type(SETepoches) is int and SETepoches > 0 and SETepoches < self._maxepoches
        self._SETepoches = SETepoches


    def set_conserveMass(self, conserveMass):
        """
        Set the flag for the use of a lambda layer to conserve mass.
        @author: Markus Pfeil
        """
        assert type(conserveMass) is bool
        self._conserveMass = conserveMass
        if self._conserveMass:
            self._initConserveMass()


    def _init_weights(self):
        """
        Initialize the weights of the ann.
        @author: Markus Pfeil
        """
        for i in range(1, len(self._unitsPerLayer)):
            [self._noParameters[ANN_Constants.ANN_LAYER_NAME.format(i)], self._weightsMask[ANN_Constants.ANN_LAYER_NAME.format(i)]] = self._createWeightsMask(self._unitsPerLayer[i-1], self._unitsPerLayer[i])
            self._weights[ANN_Constants.ANN_LAYER_NAME.format(i)] = None


    def _find_first_pos(self, array, value):
        """
        @author: Mocanu, Decebal Constantin
        """
        idx = (np.abs(array - value)).argmin()
        return idx


    def _find_last_pos(self, array, value):
        """
        @author: Mocanu, Decebal Constantin
        """
        idx = (np.abs(array - value))[::-1].argmin()
        return array.shape[0] - idx


    def _createWeightsMask(self, noRows, noCols):
        """
        Generate an Erdos Renyi sparse weights mask
        @author: Mocanu, Decebal Constantin
        """
        mask_weights = np.random.rand(noRows, noCols)
        prob = 1 - (self._epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
        mask_weights[mask_weights < prob] = 0
        mask_weights[mask_weights >= prob] = 1
        noParameters = int(np.sum(mask_weights))
        logging.info("Create Sparse Matrix: No parameters {:d}, NoRows {:d}, NoCols {:d}".format(noParameters, noRows, noCols))
        return [noParameters, mask_weights]


    def create_model(self, unitsPerLayer=[10, 100, 500, 1000], activation=['elu', 'elu', 'elu', 'elu', 'relu'], dropout=0.3):
        """
        Create model of the network
        'unitsPerLayer' includes only the hidden layer. 'activation' includes also an activation for the output layer.
        @author: Markus Pfeil
        """
        assert type(unitsPerLayer) is list
        assert type(activation) is list
        assert len(unitsPerLayer) + 1 == len(activation)  #Last activation function is for the output layer
        assert dropout is None or type(dropout) is float and 0.0 < dropout and dropout < 1.0

        self._unitsPerLayer = unitsPerLayer.copy()                                                    # Includes only the hidden layer
        self._unitsPerLayer.append(Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel]) # Include the output layer
        self._activation = activation.copy()
        self._dropout = dropout

        self._init_weights()


    def _create_model(self):
        """
        Create model of the network
        @author: Markus Pfeil
        """
        self._model = Sequential()
        self._model.add(Dense(self._unitsPerLayer[0], input_dim=Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[self._metos3dModel], name=ANN_Constants.ANN_LAYER_NAME.format(0), activation=self._activation[0]))

        for i in range(1, len(self._unitsPerLayer)):
            if self._dropout is not None:
                self._model.add(Dropout(self._dropout))
            self._model.add(Dense(self._unitsPerLayer[i], name=ANN_Constants.ANN_LAYER_NAME.format(i), kernel_constraint=MaskWeights(self._weightsMask[ANN_Constants.ANN_LAYER_NAME.format(i)]), weights=self._weights[ANN_Constants.ANN_LAYER_NAME.format(i)], activation=self._activation[i]))

        if self._conserveMass:
            #Add a layer correcting the predicted tracer concentration to recieve the correct overall mass
            self._model.add(Lambda(self._conserveMassFunction))


    def loadAnn(self):
        """
        Load the ann build with the SET algorithm from file
        @author: Markus Pfeil
        """
        self.loadAnn(custom_objects=None)


    def loadAnn(self, custom_objects=None):
        """
        Load the ann build with the SET algorithm from file
        @author: Markus Pfeil
        """
        assert custom_objects is None or type(custom_objects) is dict

        if custom_objects is None:
            custom_objects={'MaskWeights': MaskWeights, 'ConserveMass': ConserveMass, '_conserveMassFunction': self._conserveMassFunction}

        # Model reconstruction from JSON file
        filenameArchitecture = os.path.join(self._path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(self._annNumber))
        assert os.path.exists(filenameArchitecture) and os.path.isfile(filenameArchitecture)
        with open(filenameArchitecture, 'r') as f:
            self._model = model_from_json(f.read(), custom_objects=custom_objects)
        
        # Load weights into the new model
        filenameWeights = os.path.join(self._path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(self._annNumber))
        assert os.path.exists(filenameWeights) and os.path.isfile(filenameWeights)
        self._model.load_weights(filenameWeights)


    def saveAnn(self):
        """
        Save the ann
        @author: Markus Pfeil
        """
        # Save the weights
        filenameWeights = os.path.join(self._path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(self._annNumber))
        assert not os.path.exists(filenameWeights)
        self._model.save_weights(filenameWeights)
        # Save the model architecture
        filenameArchitecture = os.path.join(self._path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(self._annNumber))
        assert not os.path.exists(filenameArchitecture)
        with open(filenameArchitecture, 'w') as f:
            f.write(self._model.to_json())


    def getLoss(self):
        """
        Get the loss value of the training
        @author: Markus Pfeil
        """
        return self._losses_per_epoch['loss'][-1]


    def _rewireMask(self, weights, noWeights):
        """
        Rewire weight matrix
        @author: Mocanu, Decebal Constantin
        """
        # Remove zeta largest negative and smallest positive weights
        values = np.sort(weights.ravel())
        firstZeroPos = self._find_first_pos(values, 0)
        lastZeroPos = self._find_last_pos(values, 0)
        largestNegative = values[int((1 - self._zeta) * firstZeroPos)]
        smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self._zeta * (values.shape[0] - lastZeroPos)))]
        rewiredWeights = weights.copy()
        rewiredWeights[rewiredWeights > smallestPositive] = 1
        rewiredWeights[rewiredWeights < largestNegative] = 1
        rewiredWeights[rewiredWeights != 1] = 0
        weightMaskCore = rewiredWeights.copy()

        # Add zeta random weights
        nrAdd = 0
        noRewires = noWeights - int(np.sum(rewiredWeights))
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

        return [rewiredWeights, weightMaskCore]


    def _weightsEvolution(self):
        """
        This function represents the core of the SET procedure.
        Remove the weights closest to zero in each layer and add new random weights
        @author: Markus Pfeil
        """
        for i in range(1, len(self._unitsPerLayer)):
            layerName = ANN_Constants.ANN_LAYER_NAME.format(i)
            self._weights[layerName] = self._model.get_layer(layerName).get_weights()
            [self._weightsMask[layerName], weightMaskCore] = self._rewireMask(self._weights[layerName][0], int(self._noParameters[layerName]))
            self._weights[layerName][0] = self._weights[layerName][0] * weightMaskCore


    def train(self):
        """
        Train the artificial neural networtk
        @author: Markus Pfeil
        """
        # Read training and test data
        (x_train, x_test, y_train, y_test) = self._read_data()
        
        # Set the monitoring
        self._losses_per_epoch = {'loss': [], 'val_loss': []}
        for metric in self._metrics:
            self._losses_per_epoch[metric] = []
            self._losses_per_epoch['val_{}'.format(metric)] = []
        
        # Change the architecture after the following epochs
        epoches = list(range(self._SETepoches, self._maxepoches, self._SETepoches))
        if not self._maxepoches in epoches:
            epoches[-1] = self._maxepoches
       
        # Traing process
        for epoch in epoches:

            # Set initial epoch
            if len(epoches) >= 2 and epoch == self._maxepoches:
                initial_epoch = epoches[-2]
            elif len(epoches) == 1:
                initial_epoch = 0
            else:
                initial_epoch = epoch - self._SETepoches
            
            # Add randomly new weights
            if (epoch > self._SETepoches and len(epoches) > 1):
                self._weightsEvolution()
                K.clear_session()

            self._create_model()
            sgd = optimizers.SGD(lr=self._learning_rate, momentum=self._momentum, nesterov=self._nesterov)
            self._model.compile(loss=self._loss, optimizer=sgd, metrics=self._metrics)
            
            if epoch == self._SETepoches or len(epoches) == 1:
                self._model.summary()
            
            historytemp = self._model.fit_generator(TracerSequence(x_train, y_train, self._batch_size), steps_per_epoch=x_train.shape[0]//self._batch_size, epochs=epoch, validation_data=(x_test, y_test), initial_epoch=initial_epoch)
            
            # Monitoring the training step
            for metric in self._losses_per_epoch:
                self._losses_per_epoch[metric].append(historytemp.history[metric][0])
        
        for metric in self._losses_per_epoch:
            self._losses_per_epoch[metric] = np.asarray(self._losses_per_epoch[metric])


    def _readBoxVolumes(self, path=os.path.join(Metos3d_Constants.METOS3D_PATH, 'data', 'data', 'TMM', '2.8', 'Geometry', 'normalizedVolumes.petsc'), tracer_length=Metos3d_Constants.METOS3D_VECTOR_LEN):
        """
        Read volumes of the boxes
        @author: Markus Pfeil
        """
        assert tracer_length > 0
        assert os.path.exists(path) and os.path.isfile(path)
        
        f = open(path, 'rb')
        #Jump over the header
        np.fromfile(f, dtype='>i4', count=2)
        normvol = np.fromfile(f, dtype='>f8', count=tracer_length)
        return normvol


    def _initConserveMass(self):
        """
        Load the box volumes and calculate the overall mass of the standard initial value
        @author: Markus Pfeil
        """
        #Volume of the boxes
        filenameVolumes = os.path.join(Metos3d_Constants.METOS3D_PATH, 'data', 'data', 'TMM', '2.8', 'Geometry', 'volumes.petsc')
        vol = petsc.readPetscFile(filenameVolumes)
        assert len(vol) == Metos3d_Constants.METOS3D_VECTOR_LEN
        self._vol_vec = np.empty(shape=(Metos3d_Constants.METOS3D_MODEL_OUTPUT_LENGTH[self._metos3dModel]))
        for i in range(len(Metos3d_Constants.METOS3D_MODEL_TRACER[self._metos3dModel])):
            self._vol_vec[i*Metos3d_Constants.METOS3D_VECTOR_LEN:(i+1)*Metos3d_Constants.METOS3D_VECTOR_LEN] = vol

        #Mass of the initial tracer concentration/prediction using the ANN
        self._overallMass = sum(Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel]) * np.sum(vol)


    def _conserveMassFunction(self, tensor):
        """
        Adapt the mass of the tracer concentration tensor to the overallMass
        @author: Markus Pfeil
        """
        #Set negative concentration values to zero
        tracer_concentration = K.relu(tensor)

        #Mass of the tracer concentration/prediction using the ANN
        massPrediction = K.sum(tracer_concentration * self._vol_vec)

        #Adjust the mass
        tracer_concentration = (self._overallMass/massPrediction) * tracer_concentration

        return tracer_concentration

