#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import numpy as np
import os
import sqlite3

import metos3dutil.database.constants as DB_Constants
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


class AbstractClassDatabaseMetos3d(ABC):
    """
    Abstract class for the database access.
    @author: Markus Pfeil
    """

    def __init__(self, dbpath):
        """
        Initialization of the database connection
        @author: Markus Pfeil
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)

        self._conn = sqlite3.connect(dbpath, timeout=DB_Constants.timeout)
        self._c = self._conn.cursor()


    def close_connection(self):
        """
        Close the database connection
        @author: Markus Pfeil
        """
        self._conn.close()


    def exists_parameter(self, parameter, metos3dModel):
        """
        Get parameterId for given model and parameter values.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list

        modelParameter = Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        assert len(modelParameter) == len(parameter)

        if metos3dModel == 'MITgcm-PO4-DOP':
            parameter = [parameter[5], parameter[1], parameter[3], parameter[4], parameter[2], parameter[0], parameter[6]]

        sqlcommand = 'SELECT parameterId FROM Parameter WHERE ' + modelParameter[0] + ' = ?'
        for i in range(1, len(parameter)):
            sqlcommand = sqlcommand + ' AND ' + modelParameter[i] + ' = ?'
        self._c.execute(sqlcommand, tuple(parameter))
        parameterId = self._c.fetchall()
        return len(parameterId) > 0


    def get_parameterId(self, parameter, metos3dModel):
        """
        Get parameterId for given model and parameter values.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list

        modelParameter = Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        assert len(modelParameter) == len(parameter)

        if metos3dModel == 'MITgcm-PO4-DOP':
            parameter = [parameter[5], parameter[1], parameter[3], parameter[4], parameter[2], parameter[0], parameter[6]]

        sqlcommand = 'SELECT parameterId FROM Parameter WHERE ' + modelParameter[0] + ' = ?'
        for i in range(1, len(parameter)):
            sqlcommand = sqlcommand + ' AND ' + modelParameter[i] + ' = ?'
        self._c.execute(sqlcommand, tuple(parameter))
        parameterId = self._c.fetchall()
        assert len(parameterId) > 0 #TODO After reorg of the SBO database: len(parameterId) == 1
        return parameterId[0][0]


    def get_parameter(self, parameterId, metos3dModel):
        """
        Get parameter values of the given model for a given parameterId.
        @author: Markus Pfeil
        """
        assert type(parameterId) is int and parameterId in range(LHS_Constants.PARAMETERID_MAX+1)
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        sqlcommand = 'SELECT * FROM Parameter WHERE parameterId = ?'
        self._c.execute(sqlcommand, (parameterId, ))
        para = self._c.fetchall()
        assert len(para) == 1
        modelParameterAll = para[0][1:]
        modelParameter = np.array(modelParameterAll)[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        parameter = []
        for i in range(len(modelParameter)):
            if modelParameter[i] is not None:
                parameter.append(modelParameter[i])

        if metos3dModel == 'MITgcm-PO4-DOP':
            assert len(parameter) == 7
            parameter = [parameter[5], parameter[1], parameter[4], parameter[2], parameter[3], parameter[0], parameter[6]]

        assert len(parameter) == len(Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]])
        return np.array(parameter)


    def insert_parameter(self, parameter, metos3dModel):
        """
        Insert parameter values for the given model
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list and len(parameter) == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[metos3dModel]

        if self.exists_parameter(parameter, metos3dModel):
            #Parameter exists already in the database
            parameterId = self.get_parameterId(parameter, metos3dModel)
        else:
            #Insert parameter into the database
            sqlcommand = 'SELECT MAX(parameterId) FROM Parameter'
            self._c.execute(sqlcommand)
            dataset = self._c.fetchall()
            assert len(dataset) == 1
            parameterId = dataset[0][0] + 1

            if metos3dModel == 'MITgcm-PO4-DOP':
                parameter = [parameter[5], parameter[1], parameter[3], parameter[4], parameter[2], parameter[0], parameter[6]]

            purchases = []
            modelParameter = [None for _ in range(20)]
            i = 0
            for j in range(20):
                if Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel][j]:
                    modelParameter[j] = parameter[i]
                    i = i + 1
            assert i == Metos3d_Constants.METOS3D_MODEL_INPUT_PARAMTER_LENGTH[metos3dModel]

            purchases.append((parameterId,) + tuple(modelParameter))
            self._c.executemany('INSERT INTO Parameter VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
            self._conn.commit()

        return parameterId

