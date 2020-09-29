#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import os
import pyDOE

import metos3dutil.metos3d.constants as Metos3d_Constants


class LatinHypercubeSample():
    """
    Create and read the latin hypercube sample for the model parameter for the hierarchy of the biogeochemical models
    @author: Markus Pfeil
    """

    def __init__(self, lhs_filename, noParameters=20, samples=100, criterion=None):
        """
        Initialisation of the latin hypercube sample
        @author: Markus Pfeil
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
        @author: Markus Pfeil
        """
        assert np.shape(lowerBoundary) == np.shape(lb)
        self._lb = lowerBoundary


    def set_upperBoundary(self, upperBoundary):
        """
        Set the upper boundary for the parameter values
        @author: Markus Pfeil
        """
        assert np.shape(upperBoundary) == np.shape(ub)
        self._ub = upperBoundary


    def create(self):
        """
        Create the latin hypercube sample
        @author: Markus Pfeil
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
        @author: Markus Pfeil
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
        @author: Markus Pfeil
        """
        fid = open(self._lhs_filename, 'rb')
        self._samples, = np.fromfile(fid, dtype = '>i4', count = 1)
        self._noParameters, = np.fromfile(fid, dtype = '>i4', count = 1)
        x = np.fromfile(fid, dtype = '>f8', count = self._samples * self._noParameters)
        fid.close()
        
        self._lhs = np.reshape(x, (self._noParameters, self._samples))


    def get_sample_count(self):
        """
        Get the number of samples
        @author: Markus Pfeil
        """
        return self._samples


    def get_all_parameter(self, lhs_index):
        """
        Get all model parameters for the given index of the latin hypercube sample
        @author: Markus Pfeil
        """
        assert type(lhs_index) is int and 0 <= lhs_index and lhs_index < self._samples

        return self._lhs[:, lhs_index]


    def get_parameter(self, model, lhs_index):
        """
        Get the model parameter for the given model and index of the latin hypercube sample
        @author: Markus Pfeil
        """
        assert model in Metos3d_Constants.METOS3D_MODELS
        assert type(lhs_index) is int and 0 <= lhs_index and lhs_index < self._samples

        return self.get_all_parameter(lhs_index)[Metos3d_Constants.PARAMETER_RESTRICTION[model]]

