#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os
import pyDOE

import metos3dutil.metos3d.constants as Metos3d_Constants


class LatinHypercubeSample():
    """
    Latin hypercube sample of model parameter of a biogeochemical models
    """

    def __init__(self, lhs_filename, noParameters=20, samples=100, criterion=None):
        """
        Initialisation of the latin hypercube sample

        Initialisation of the latin hypercube sample for the model parameter of the hierarchy of biogeochemical models

        Parameters
        ----------
        lhs_filename : str
            Filename of the latin hypercube sample
        noParameters : int, default: 20
            Number of parameter of the latin hypercube sample
        samples : int, default: 100
            Number of samples in the latin hypercube sample
        criterion : {None, 'center', 'maximin', 'centermaximin', 'correlation'}
            Criterion how to sample the points

        Attributes
        ----------
        _lb : numpy.ndarray
            Numpy array with the lower bounds of the model parameter
        _up : numpy.ndarray
            Numpy array with the upper bounds of the model parameter
        """
        assert type(noParameters) is int and 0 < noParameters
        assert type(samples) is int and 0 < samples
        assert criterion in [None, 'center', 'maximin', 'centermaximin', 'correlation']

        self._lhs_filename = lhs_filename   # File name of the latin hypercube sample
        self._noParameters = noParameters   # Number of the model parameters (for the whole model hierarchy)
        
        self._samples = samples             # Number of samples
        self._criterion = criterion         # Criterion how to sample the points (None, center, maximin, centermaximin or correlation)
        
        # Set initial boundaries for the lhs samples
        self._lb = Metos3d_Constants.LOWER_BOUND
        self._ub = Metos3d_Constants.UPPER_BOUND
        
        if os.path.exists(lhs_filename) and os.path.isfile(lhs_filename):
            self.read_Lhs()
        else:
            self._lhs = None


    def set_lowerBoundary(self, lowerBoundary):
        """
        Set the lower boundary for the parameter values

        Parameters
        ----------
        lowerBoundary : numpy.ndarray
            Numpy array with lower boundaries of each model parameter

        Raises
        ------
        AssertionError
            If the shape of lowerBoundary array does not match the count of
            model parameter in the latin hypercube sample
        """
        assert np.shape(lowerBoundary) == np.shape(lb)
        self._lb = lowerBoundary


    def set_upperBoundary(self, upperBoundary):
        """
        Set the upper boundary for the parameter values

        Parameters
        ----------
        upperBoundary : numpy.ndarray
            Numpy array with upper boundaries of each model parameter

        Raises
        ------
        AssertionError
            If the shape of upperBoundary array does not match the count of
            model parameter in the latin hypercube sample
        """
        assert np.shape(upperBoundary) == np.shape(ub)
        self._ub = upperBoundary


    def create(self):
        """
        Create the latin hypercube sample

        Create the latin hypercube sample using the function lhs of the package
        pyDOE
        """
        assert (self._noParameters,) == np.shape(self._lb)
        assert (self._noParameters,) == np.shape(self._ub)
        lhs = pyDOE.lhs(len(self._lb), samples=self._samples, criterion=self._criterion)
        self._lhs = np.zeros(shape=np.shape(lhs), dtype = '>f8')
        self._lhs = self._lb + (self._ub - self._lb) * lhs
        self._lhs = self._lhs.transpose()
        
        
    def write_Lhs(self):
        """
        Write latin hypercube samples in file with big endian order

        Raises
        ------
        AssertionError
            If a file with the set filename of the latin hypercube sample
            already exists.
        """
        samples = np.array(self._samples, dtype = '>i4')
        noParameters = np.array(self._noParameters, dtype = '>i4')
        
        assert not os.path.exists(self._lhs_filename)
        fid = open(self._lhs_filename, 'wb')
        samples.tofile(fid)
        noParameters.tofile(fid)
        for i in range(self._noParameters):
            for j in range(self._samples):
                value = np.array(self._lhs[i,j], dtype = '>f8')
                value.tofile(fid)
        fid.close()


    def read_Lhs(self):
        """
        Read latin hypercube sample out file in big endian order

        Raises
        ------
        AssertionError
            If the file with the set filename of the latin hypercube sample
            does not exist.
        """
        assert os.path.exists(self._lhs_filename)
        fid = open(self._lhs_filename, 'rb')
        self._samples, = np.fromfile(fid, dtype = '>i4', count = 1)
        self._noParameters, = np.fromfile(fid, dtype = '>i4', count = 1)
        x = np.fromfile(fid, dtype = '>f8', count = self._samples * self._noParameters)
        fid.close()
        
        self._lhs = np.reshape(x, (self._noParameters, self._samples))


    def get_sample_count(self):
        """
        Returns the number of samples

        Returns
        -------
        int
            Number of samples
        """
        return self._samples


    def get_all_parameter(self, lhs_index):
        """
        Returns all model parameters

        Returns all model parameters of the hierarchy for the given index

        Parameters
        ----------
        lhs_index : int
            Id of the parameter of the latin hypercube example

        Raises
        ------
        AssertionError
            If the lhs_index is greater or equal than the number of samples.
        """
        assert type(lhs_index) is int and 0 <= lhs_index and lhs_index < self._samples

        return self._lhs[:, lhs_index]


    def get_parameter(self, metos3dModel, lhs_index):
        """
        Returns the model parameter for the given model

        Returns the model parameter for the given model of the hierarchy for
        the given index

        Parameters
        ----------
        metos3dModel : str
            Name of the biogeochemical model
        lhs_index : int
            Id of the parameter of the latin hypercube example

        Raises
        ------
        AssertionError
            If the lhs_index is greater or equal than the number of samples.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(lhs_index) is int and 0 <= lhs_index and lhs_index < self._samples

        return self.get_all_parameter(lhs_index)[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]

