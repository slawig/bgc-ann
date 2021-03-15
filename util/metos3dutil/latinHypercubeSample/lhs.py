#!/usr/bin/env python
# -*- coding: utf8 -*

import os
from metos3dutil.latinHypercubeSample.LatinHypercubeSample import LatinHypercubeSample as LatinHypercubeSample
import metos3dutil.latinHypercubeSample.constants as LHS_Constants
import metos3dutil.metos3d.constants as Metos3d_Constants


def readParameterValues(parameterId, metos3dModel):
    """
    Returns model parameter of the latin hypercube sample

    Returns the model parameter the latin hypercube sample for the given
    biogeochemical model and parameterId

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
        If the binary files of the latin hypercube sample does not exist.
    """
    assert parameterId in range(0, LHS_Constants.PARAMETERID_MAX+1)
    assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
    
    if parameterId == 0:
        p = Metos3d_Constants.REFERENCE_PARAMETER
        return p[Metos3d_Constants.PARAMETER_RESTRICTION[metos3dModel]]
    elif parameterId <= 100:
        #TODO: Use importlib resources to import the files in the package and remoew LHS_PATH
        lhsFilename = os.path.join(LHS_Constants.LHS_PATH, LHS_Constants.FILENAME_LHS_100)
        assert os.path.exists(lhsFilename) and os.path.isfile(lhsFilename)
        lhs = LatinHypercubeSample(lhsFilename, samples=100)
        return lhs.get_parameter(metos3dModel, int(parameterId-1))
    elif parameterId <= 1100:
        lhsFilename = os.path.join(LHS_Constants.LHS_PATH, LHS_Constants.FILENAME_LHS_1000)
        assert os.path.exists(lhsFilename) and os.path.isfile(lhsFilename)
        lhs = LatinHypercubeSample(lhsFilename, samples=1000)
        return lhs.get_parameter(metos3dModel, int(parameterId-101))
    else:
        lhsFilename = os.path.join(LHS_Constants.LHS_PATH, LHS_Constants.FILENAME_LHS_10000)
        assert os.path.exists(lhsFilename) and os.path.isfile(lhsFilename)
        lhs = LatinHypercubeSample(lhsFilename, samples=10000)
        return lhs.get_parameter(metos3dModel, int(parameterId-1101))

