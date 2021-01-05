#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
import os
import threading

import ann.network.constants as ANN_Constants
from ann.database.access import Ann_Database


class AbstractClassData(ABC):
    """
    Abstract class for the data of an ANN.
    @author: Markus Pfeil
    """

    def __init__(self, annId):
        """
        Initialization of the evaluation class.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)

        #Logging
        self.queue = mp.Queue()
        self.logger = logging.getLogger(__name__)
        self.lp = threading.Thread(target=self.logger_thread)

        #Database
        self._annDatabase = Ann_Database()

        #Artificial neural network
        self._annId = annId
        self._getAnnConfig()

        #Path
        self._pathPrediction = os.path.join(ANN_Constants.PATH, 'Prediction')
        self._path = os.path.join(self._pathPrediction, self._model, 'AnnId_{:0>5d}'.format(self._annId))

        #Metos3dModel
        self._timestep = 1


    def logger_thread(self):
        """
        Logging for multiprocessing.
        @author: Markus Pfeil
        """
        while True:
            record = self.queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)


    def close_DB_connection(self):
        """
        Close the database connection.
        @author: Markus Pfeil
        """
        self._annDatabase.close_connection()


    def _getAnnConfig(self):
        """
        Read the configuration if the ANN from the database.
        @author: Markus Pfeil
        """
        self._annType, self._annNumber, self._model, self._annConverseMass = self._annDatabase.get_annTypeNumberModelMass(self._annId)


    def _getModelParameter(self, parameterId):
        """
        Read model parameter for the given parameter id and model.
        @author: Markus Pfeil
        """
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)

        self._modelParameter = list(self._annDatabase.get_parameter(parameterId, self._model))

