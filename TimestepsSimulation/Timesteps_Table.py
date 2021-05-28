#!/usr/bin/env python
# -*- coding: utf8 -*

import os

import metos3dutil.metos3d.constants as Metos3d_Constants
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsDatabase import Timesteps_Database


def main():
    """
    Create latex table string using different time steps for the spin up

    Create latex table strings of the results using different time steps for
    the spin up calculation.

    Parameters
    ----------
    orientation : str
        Orientation of the figure
    fontsize : int
        Fontsize used in the figure
    plotSpinup : bool, default: False
        If True, plot the figure of the spin up norm
    plotNorm : bool, default: False
        If True, plot the figures for the norm
    """
    timestepTable = TimestepsTables()

    #Table with the number of oscillations
#    oscillationCount = timestepTable.oscillationCount()
#    print('Table with the number of oscillations:\n{:s}'.format(oscillationCount))

    #Table with the distribution of oscillations
#    oscillationDistribution = timestepTable.oscillationDistribution()
#    print('Table with oscillation distribution:\n{:s}'.format(oscillationDistribution))

    #Table with the costfunction value for the reference parameter
    costfunction = timestepTable.costfunction()
    print('Table with cost function values:\n{:s}'.format(costfunction))

    timestepTable.closeDatabaseConnection()



class TimestepsTables():
    """
    Preparation of latex tables for the results using different time steps

    Preparation of latex table strings for the results using different time
    steps for the spin up calculation.
    """

    def __init__(self, dbpath=Timesteps_Constants.DB_PATH, cmap=None, completeTable=True):
        """
        Constructs the environment to create latex table strings of the data
        using different time steps for the spin up calculation.

        Parameter
        ----------
        dbpath : str, default: timesteps.constants.DB_PATH
            Path to the sqlite database
        completeTable : bool, default: True
            If the value is True, use all columns (even columns with value
            None) in SELECT queries on the database
        """
        assert os.path.exists(dbpath) and os.path.isfile(dbpath)
        assert type(completeTable) is bool

        self._database = Timesteps_Database(dbpath=dbpath, completeTable=completeTable)


    def closeDatabaseConnection(self):
        """
        Close the connection of the database
        """
        self._database.close_connection()


    def costfunction(self, parameterId=0, year=10000, costfunction='OLS', measurementId=0):
        """
        Table with the cost function values for each model and time step

        Create a latex table string with the cost function values using the
        spin-up result after the given model years. The table includes this
        values for each time step and every biogeochemical model.

        PARAMETER
        ---------
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        year : int, default: 10000
            Model year of the spin-up calculation
        costfunction : {'OLS', 'GLS', 'WLS'}, default: 'OLS'
            Used cost function
        measurementId : int, default: 0
            Selection of the tracer included in the cost function calculation

        RETURNS
        -------
        str
            Latex table string with the cost function value for the
            different biogeochemical models and time steps
        """
        assert parameterId in range(0, Timesteps_Constants.PARAMETERID_MAX+1)
        assert type(year) is int and 0 <= year
        assert costfunction in ['OLS', 'GLS', 'WLS']
        assert type(measurementId) is int and 0 <= measurementId

        tableStr = '\\hline\nTime step & {:s}  \\\\\n\\hline\n'.format(' & '.join(map(str, Metos3d_Constants.METOS3D_MODELS)))
        for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
            tableStr = tableStr + '{:>2d}dt'.format(timestep)
            for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                costfunctionValue = self._database.get_table_costfunction_value(parameterId=parameterId, model=metos3dModel, timestep=timestep, year=year, costfunction=costfunction, measurementId=measurementId)
                if len(costfunctionValue) == 1:
                    tableStr = tableStr + ' & {:.3e}'.format(costfunctionValue[0][1])
                else:
                    tableStr = tableStr + ' & {:^9s}'.format('-')
            tableStr = tableStr + ' \\\\\n'
        tableStr = tableStr + '\\hline\n'

        return tableStr


    def oscillationCount(self):
        """
        Table with number of oscillations of the spin up norm

        Create latex table string with the number of oscillations of the spin
        up norm for the different model parameters. The table includes this
        number for the different biogeochemical models.

        RETURNS
        -------
        str
            Latex table string with the number of oscillations for the
            different biogeochemical models and time steps
        """
        tableStr = 'Time step & {:s}  \\\\\n'.format(' & '.join(map(str, Metos3d_Constants.METOS3D_MODELS)))
        for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
            tableStr = tableStr + '{:>2d}'.format(timestep)
            for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                #List of simulationIds for the model and time step
                simulationIdList = self._database.get_simids_for_model_timestep(metos3dModel, timestep)

                #Number of oscillations
                oscillationCount = 0
                for simulationId in simulationIdList:
                    if self._database.oscillation_spin(simulationId):
                        oscillationCount = oscillationCount + 1

                tableStr = tableStr + ' & {:>3d}'.format(oscillationCount)

            tableStr = tableStr + ' \\\\\n'
        return tableStr


    def oscillationDistribution(self):
        """
        Table with the distribution of the oscillations

        Table with the distribution of the oscillations to the different
        combinations of time steps for all parameters.

        RETURNS
        -------
        str
            Latex table string with the distribution of the oscillations
        """
        concentrationIdDic = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}
        oscillationDistribution = {}
        keyDic = {}

        for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
            oscllationDistributionDic = {}
            for parameterId in range(Timesteps_Constants.PARAMETERID_MAX+1):
                oscillationDic = {1: '', 2: '', 4: '', 8: '', 16: '', 32: '', 64: ''}
                for timestep in Metos3d_Constants.METOS3D_TIMESTEPS:
                    simulationId = self._database.get_simulationId(metos3dModel, parameterId, concentrationIdDic[metos3dModel], timestep=timestep)
                    if self._database.get_convergence(simulationId):
                        oscillationDic[timestep] = self._database.oscillation_spin(simulationId)

                countTimesteps = 0
                keyStr = ''
                for key in oscillationDic:
                    if oscillationDic[key]:
                        keyStr = keyStr + '{}'.format(key)
                        countTimesteps += 1

                if keyStr in oscllationDistributionDic:
                    oscllationDistributionDic[keyStr] += 1
                else:
                    oscllationDistributionDic[keyStr] = 1

                if keyStr not in keyDic:
                    keyDic[keyStr] = countTimesteps

            oscillationDistribution[metos3dModel] = oscllationDistributionDic
            print(oscllationDistributionDic)

        #Table
        tableStr = '& {:s}  \\\\\n'.format(' & '.join(map(str, Metos3d_Constants.METOS3D_MODELS)))
        for keyStr in keyDic:
            tableStr = tableStr + '{:s}'.format(keyStr)
            for metos3dModel in Metos3d_Constants.METOS3D_MODELS:
                tableStr += ' & {:>2}'.format(oscillationDistribution[metos3dModel][keyStr]) if keyStr in oscillationDistribution[metos3dModel] else ' &  0'
            tableStr += ' \\\\\n'

        return tableStr


if __name__ == '__main__':
    main()

