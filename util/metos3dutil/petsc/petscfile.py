#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import numpy as np


PETSC_VEC_HEADER = 1211214


def readPetscFile(filename):
    """
    Returns a numpy array of a readed petsc vector of the tracer concentration

    Parameters
    ----------
    filename : str
        Filename of the petsc file

    Returns
    -------
    numpy.ndarray
        Numpy array of the tracer concentration

    Raises
    ------
    AssertionError
        If the petsc file does not exist.
    """
    assert os.path.exists(filename) and os.path.isfile(filename)

    with open(filename, 'r') as f:
        #Omit header
        vecid = np.fromfile(f, dtype='>i4', count=1)
        #Read length
        nvec = np.fromfile(f, dtype='>i4', count=1)
        assert nvec.ndim == 1 and len(nvec) == 1
        nvec = nvec[0]
        #Read values
        v = np.fromfile(f, dtype='>f8', count=nvec)
        assert v.ndim == 1 and len(v) == nvec

    return v


def writePetscFile(filename, vec):
    """
    Save a numpy vector as petsc file

    Parameters
    ----------
    filename : str
        Filename of the petsc file
    vec : numpy.ndarray
        Numpy array with the tracer concentration
    """
    assert type(vec) is np.ndarray
 
    with open(filename, mode='xb') as f:
        #Write header
        header = np.array(PETSC_VEC_HEADER, dtype='>i4')
        header.tofile(f)

        #Write length (32 bit int)
        length = np.array(len(vec), dtype='>i4')
        length.tofile(f)

        #Write values
        vec = vec.astype('>f8')
        vec.tofile(f)


def readPetscMatrix(filename):
    """
    Returns a numpy array of a readed petsc matrix

    Parameters
    ----------
    filename : str
        Filename of the petsc file

    Returns
    -------
    numpy.ndarray
        Numpy array of the matrix

    Raises
    ------
    AssertionError
        If the petsc file does not exist.
    """
    assert os.path.exists(filename) and os.path.isfile(filename)

    with open(filename, 'rb') as f:
        #Omit header
        vecid = np.fromfile(f, dtype='>i4', count=1)
        #Read dims
        nx = np.fromfile(f, dtype='>i4', count=1)[0]
        ny = np.fromfile(f, dtype='>i4', count=1)[0]
        nnz = np.fromfile(f, dtype='>i4', count=1)[0]
        nrow = np.fromfile(f, dtype='>i4', count=nx)
        colidx = np.fromfile(f, dtype='>i4', count=nnz)
        val = np.fromfile(f, dtype='>f8', count=nnz)
        
    # create full matrix
    matfull = np.zeros(shape=(nx, ny), dtype = float)
    offset = 0
    for i in range(nx):
        if not nrow[i] == 0.0:
            for j in range(nrow[i]):
                matfull[i, colidx[offset]] = val[offset]
                offset = offset + 1

    return matfull

