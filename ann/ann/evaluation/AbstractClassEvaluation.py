#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import os
import multiprocessing as mp
import threading
import logging

import ann.network.constants as ANN_Constants
from ann.database.access import Ann_Database


class AbstractClassEvaluation(ABC):
    """
    Abstract class for the evaluation of an ANN.
    @author: Markus Pfeil
    """

    def __init__(self, annId, parameterId, massAdjustment=False, tolerance=None, spinupToleranceReference=False):
        """
        Initialization of the evaluation class.
        @author: Markus Pfeil
        """
        assert annId in range(0, ANN_Constants.ANNID_MAX+1)
        assert parameterId in range(0, ANN_Constants.PARAMETERID_MAX+1)
        assert type(massAdjustment) is bool
        assert tolerance is None or (type(tolerance) is float and tolerance > 0)
        assert type(spinupToleranceReference) is bool and ((not spinupToleranceReference) or (spinupToleranceReference and tolerance > 0))

        #Logging
        self.queue = mp.Queue()
        self.logger = logging.getLogger(__name__)
        self.lp = threading.Thread(target=self.logger_thread)

        #Database
        self._annDatabase = Ann_Database()

        #Artificial neural network
        self._annId = annId
        self._getAnnConfig()
        self._massAdjustment = massAdjustment
        self._tolerance = tolerance if (tolerance is not None) else 0.0
        self._simulationPath = self._setSimulationPath()

        #Metos3dModel
        self._parameterId = parameterId
        self._timestep = 1

        self._spinupTolerance = True if (tolerance is not None) else False
        self._spinupToleranceReference = spinupToleranceReference
 

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


    def _getModelParameter(self):
        """
        Read model parameter for the given parameter id and model.
        @author: Markus Pfeil
        """
        self._modelParameter = self._annDatabase.get_parameter(self._parameterId, self._model)


    def _setSimulationPath(self):
        """
        Create the simulation path.
        @author: Markus Pfeil
        """
        if self._spinupToleranceReference:
            simulationPath = os.path.join(ANN_Constants.PATH, 'SpinupTolerance', self._model, 'Tolerance_{:.1e}'.format(self._tolerance), 'Parameter_{:0>3d}'.format(self._parameterId))
        else:
            simulationPath = os.path.join(ANN_Constants.PATH, 'Prediction', self._model, 'AnnId_{:0>5d}'.format(self._annId), 'Parameter_{:0>3d}'.format(self._parameterId))
            if self._spinupTolerance and self._massAdjustment:
                simulationPath = os.path.join(simulationPath, 'SpinupTolerance', 'Tolerance_{:.1e}'.format(self._tolerance))
            elif self._spinupTolerance:
                simulationPath = os.path.join(simulationPath, 'SpinupTolerance', 'NoMassAdjustment', 'Tolerance_{:.1e}'.format(self._tolerance))
            elif self._massAdjustment:
                simulationPath = os.path.join(simulationPath, 'MassAdjustment')
        os.makedirs(simulationPath, exist_ok=True)
        return simulationPath
