#!/usr/bin/env python
# -*- coding: utf8 -*

import logging
import numpy as np
import os
import sqlite3
import time

import metos3dutil.metos3d.constants as Metos3d_Constants
from metos3dutil.metos3d.Metos3d import Metos3d
from metos3dutil.simulation.AbstractClassSimulation import AbstractClassSimulation
import timesteps.constants as Timesteps_Constants
from timesteps.TimestepsDatabase import Timesteps_Database


class TimestepsSimulation(AbstractClassSimulation):
    """
    Class for the simulation using different time steps
    """

    def __init__(self, metos3dModel, parameterId=0, timestep=1):
        """
        Initializes the simulation for different time steps

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        parameterId : int, default: 0
            Id of the parameter of the latin hypercube example
        timestep : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the spin up simulation

        Attributes
        ----------
        _defaultConcentration : bool
            If True, uses standard constant initial concentration
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(parameterId) is int and parameterId in range(Timesteps_Constants.PARAMETERID_MAX+1)
        assert timestep in Metos3d_Constants.METOS3D_TIMESTEPS

        #Time
        startTime = time.time()

        AbstractClassSimulation.__init__(self, metos3dModel, parameterId=parameterId, timestep=timestep)

        logging.info('***Initialization of TimestepsSimulation:***\nMetos3dModel: {:s}\nParameterId: {:d}\nTime step: {:d}dt\nConcentrationId: {:d}'.format(self._metos3dModel, self._parameterId, self._timestep, self._concentrationId))
        logging.info('***Time for initialization: {:.6f}s***\n\n'.format(time.time() - startTime))


    def _init_database(self):
        """
        Inits the database connection
        """
        self._database = Timesteps_Database()


    def set_concentrationId(self, concentrationId=None):
        """
        Sets the id of the initial concentration

        Parameters
        ----------
        concentrationId : int or None, default: None
            Id of the initial concentration. If None, uses the id of the
            standard constant initial concentration
        """
        assert concentrationId is None or type(concentrationId) is int and 0 <= concentrationId

        if concentrationId is None:
            self._concentrationId = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
            self._defaultConcentration = True
        else:
            self._concentrationId = concentrationId
            self._defaultConcentration = False

        self._set_simulationId()
        self._set_path()


    def _set_path(self):
        """
        Sets the path to the simulation directory
        """
        self._path = os.path.join(Timesteps_Constants.PATH, 'Timesteps', self._metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(self._timestep))
        if not self._defaultConcentration:
            self._path = os.path.join(self._path, 'InitialConcentration_{:0>3d}'.format(self._concentrationId))


    def _startSimulation(self):
        """
        Starts the spin up simulation

        Notes
        -----
        Creates the directory of the simulation
        """
        os.makedirs(self._path, exist_ok=True)

        metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._years, nodes=self._nodes)
        metos3d.setTrajectoryParameter(trajectoryYear=self._trajectoryYear)

        #Set the initial concentration for the spin up
        if not self._defaultConcentration:
            metos3d.setInitialConcentration([float(c) for c in self._database.get_concentration(self._concentrationId)[Metos3d_Constants.METOS3D_MODEL_TRACER_MASK[self._metos3dModel]]])

        if self._spinupTolerance is not None:
            metos3d.setTolerance(self._spinupTolerance)

        #Run the spin up simulation
        metos3d.run()


    def _set_calculateNormReferenceSimulationParameter(self, timestepReference=1):
        """
        Returns parameter of the norm calculation

        Parameters
        ----------
        timestepReference : {1, 2, 4, 8, 16, 32, 64}, default: 1
            Time step of the reference solution

        Returns
        -------
        tuple
            The tuple contains
              - the simulationId of the simulation used as reference
                simulation and
              - path of the directory of the reference simulation
        """
        concentrationIdReference = self._database.get_concentrationId_constantValues(self._metos3dModel, Metos3d_Constants.INITIAL_CONCENTRATION[self._metos3dModel])
        simulationIdReference = self._database.get_simulationId(self._metos3dModel, self._parameterId, concentrationIdReference, timestep=timestepReference)

        pathReferenceTracer = os.path.join(Timesteps_Constants.PATH, 'Timesteps', self._metos3dModel, 'Parameter_{:0>3d}'.format(self._parameterId), '{:d}dt'.format(timestepReference))

        return (simulationIdReference, pathReferenceTracer)


    def evaluation(self):
        """
        TODO: Funktion loeschen!
        Evaluation of the spin up simulation

        Notes
        -----
        Inserts the values of the spin up norm, the norm of the tracer
        concentration as well as the norm of the concentration difference
        using a reference solution into the database. Moreover, the values of
        the norm over the whole trajectory is calculated and inserted for the
        last model year of the spin up.
        """
        #Insert the spin up values
        if self.existsMetos3dOutput():
            self._insertSpinup()

        #Insert the tracer norm and deviation values
        if self.existsMetos3dOutput() and (not self._checkNormTotalityDatabase() or not self._checkDeviationTotalityDatabase()):
            self._calculateNorm()

        #Insert the norm of the trajectory
        if self._trajectoryFlag and self.existsMetos3dOutput() and not self._checkTrajectoryNormTotalityDatabase():
            self._calculateTrajectoryNorm()


    def _insertSpinup(self):
        """
        Inserts the spin up norm and convergence values into the database

        Reads the spin up norm values from the Metos3d job output and inserts
        these values into the database.
        """
        metos3d = Metos3d(self._metos3dModel, self._timestep, self._modelParameter, self._path, modelYears=self._years, nodes=self._nodes)
        spinupNorm = metos3d.read_spinup_norm_values()
        spinupNormShape = np.shape(spinupNorm)

        try:
            for i in range(spinupNormShape[0]):
                year = int(spinupNorm[i,0])
                tolerance = float(spinupNorm[i,1])
                norm = float(spinupNorm[i,2]) if spinupNormShape[1] == 3 else None

                if year == 0 and tolerance == 0.0 and spinupNorm is not None and norm == 0.0:
                    raise ValueError()

                #TODO: Remove comment
                self._database.insert_spinup(self._simulationId, year, tolerance, norm, overwrite=self._overwrite)

            self._database.insert_convergence(self._simulationId, True, overwrite=self._overwrite)
        except (sqlite3.IntegrityError, ValueError):
            logging.error('Inadmissable values for simulationId {:0>4d} and year {:0>4d}\n'.format(self._simulationId, year))
            self._database.insert_convergence(self._simulationId, False, overwrite=self._overwrite)


    def calculateNormSetReference(self):
        """
        Calculates the tracer norm values for every tracer output

        Instead of the standard reference solution (i.e. the simulaiton using
        the same parameterId and standard initial concentration but time step
        1dt), we use as reference solution the simulation calculated with the
        same parameterId and time step but with the standard initial
        concentration.
        """
        simulationIdReference, pathReferenceTracer = self._set_calculateNormReferenceSimulationParameter(timestepReference=self._timestep)
        self._calculateNorm(simulationIdReference=simulationIdReference, pathReferenceTracer=pathReferenceTracer)

