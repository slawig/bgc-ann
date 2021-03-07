#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import numpy as np
import os
import sqlite3
import time

import metos3dutil.database.constants as DB_Constants
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


class AbstractClassDatabaseMetos3d(ABC):
    """
    Abstract class for the database access.
    """

    def __init__(self, dbpath):
        """
        Initialization of the database connection

        Parameters
        ----------
        dbpath : str
            Path to the sqlite file of the sqlite database

        Raises
        ------
        AssertionError
            If the dbpath does not exists.
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)

        self._conn = sqlite3.connect(dbpath, timeout=DB_Constants.timeout)
        self._c = self._conn.cursor()


    def close_connection(self):
        """
        Close the database connection
        """
        self._conn.close()


    #TODO @abstractmethod
    def create_database(self):
        """
        Create all table of the database
        """
        pass


    def _create_table_parameter(self):
        """
        Create table parameter
        """
        self._c.execute('''CREATE TABLE Parameter (parameterId INTEGER NOT NULL, k_w REAL NOT NULL, k_c REAL NOT NULL, mu_P REAL NOT NULL, mu_Z REAL NOT NULL, K_N REAL NOT NULL, K_P REAL NOT NULL, K_I REAL NOT NULL, simga_Z REAL NOT NULL, sigma_DOP REAL NOT NULL, lambda_P REAL NOT NULL, kappa_P REAL NOT NULL, lambda_Z REAL NOT NULL, kappa_Z REAL NOT NULL, lambda_prime_P REAL NOT NULL, lambda_prime_Z REAL NOT NULL, lambda_prime_D REAL NOT NULL, lambda_prime_DOP REAL NOT NULL, b REAL NOT NULL, a_D REAL NOT NULL, b_D REAL NOT NULL, PRIMARY KEY (parameterId))''')


    def exists_parameter(self, parameter, metos3dModel):
        """
        Returns if the parameter exists for given model in the database

        Parameters
        ----------
        parameter : list [float]
            Model parameter of the biogeochemical model
        metos3dModel : str
            Name of the biogeochemical model

        Returns
        -------
        bool
            True if an entry exists for the given model parameter
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list

        modelParameter = Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        assert len(modelParameter) == len(parameter)

        #Different order of the model parameter for the MITgcm-PO4-DOP model
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
        Returns the parameterId for given model and parameter values

        Parameters
        ----------
        parameter : list [float]
            Model parameter of the biogeochemical model
        metos3dModel : str
            Name of the biogeochemical model

        Returns
        -------
        int
            parameterId of the given model parameter

        Raises
        ------
        AssertionError
            If no entry for the model parameter exists in the table Parameter
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameter) is list

        modelParameter = Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
        assert len(modelParameter) == len(parameter)

        #Different order of the model parameter for the MITgcm-PO4-DOP model
        if metos3dModel == 'MITgcm-PO4-DOP':
            parameter = [parameter[5], parameter[1], parameter[3], parameter[4], parameter[2], parameter[0], parameter[6]]

        sqlcommand = 'SELECT parameterId FROM Parameter WHERE ' + modelParameter[0] + ' = ?'
        for i in range(1, len(parameter)):
            sqlcommand = sqlcommand + ' AND ' + modelParameter[i] + ' = ?'
        self._c.execute(sqlcommand, tuple(parameter))
        parameterId = self._c.fetchall()
        assert len(parameterId) >= 1 #TODO == 1 after reorg of SBO_Database
        return parameterId[0][0]


    def get_parameter(self, parameterId, metos3dModel):
        """
        Returns the parameter values of the metos3dModel for the parameterId

        Parameters
        ----------
        parameterId : int
            Id of the parameter of the latin hypercube example
        metos3dModel : str
            Name of the biogeochemical model

        Returns
        -------
        numpy.ndarray
            Numpy array with the model parameter

        Raises
        ------
        AssertionError
            If no or more than one entry exists in the database
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

        #Different order of the model parameter for the MITgcm-PO4-DOP model
        if metos3dModel == 'MITgcm-PO4-DOP':
            assert len(parameter) == 7
            parameter = [parameter[5], parameter[1], parameter[4], parameter[2], parameter[3], parameter[0], parameter[6]]

        assert len(parameter) == len(Metos3d_Constants.PARAMETER_NAMES[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]])
        return np.array(parameter)


    def insert_parameter(self, parameter, metos3dModel):
        """
        Insert parameter values for the given model

        Insert the parameter values for the given model. If a database entry
        with these parameters already exists, only the parameterId is returned

        Parameters
        ----------
        parameter : list [float]
            Model parameter of the biogeochemical model
        metos3dModel : str
            Name of the biogeochemical model

        Returns
        -------
        int
            parameterId of the given model parameter

        Raises
        ------
        sqlite3.OperationalError
            If the parameter could not be successfully inserted into the
            database after serveral attempts

        Notes
        -----
        After an incorrect insert, this function waits a few seconds before the
        next try
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

            #Different order of the model parameter for the MITgcm-PO4-DOP model
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

            inserted = False
            insertCount = 0
            while(not inserted and insertCount < DB_Constants.INSERT_COUNT):
                try:
                    self._c.executemany('INSERT INTO Parameter VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', purchases)
                    self._conn.commit()
                    inserted = True
                except sqlite3.OperationalError:
                    insertCount += 1
                    #Wait for the next insert
                    time.sleep(DB_Constants.TIME_SLEEP)

        return parameterId

