#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import numpy as np

import system.system


# Metos3d
METOS3D_VECTOR_LEN = 52749
METOS3D_MODELS = ['N', 'N-DOP', 'NP-DOP', 'NPZ-DOP', 'NPZD-DOP', 'MITgcm-PO4-DOP']
METOS3D_TIMESTEPS = [1, 2, 4, 8, 16, 32, 64]
METOS3D_STEPS_PER_YEAR = 2880
METOS3D_GRID_LATITUDE = 2.8125
METOS3D_GRID_LONGITUDE = 2.8125
METOS3D_GRID_DEPTH = 15

# Model tracer
METOS3D_MODEL_TRACER = {}
METOS3D_MODEL_TRACER['N'] = ('N')
METOS3D_MODEL_TRACER['N-DOP'] = ('N', 'DOP')
METOS3D_MODEL_TRACER['NP-DOP'] = ('N', 'P', 'DOP')
METOS3D_MODEL_TRACER['NPZ-DOP'] = ('N', 'P', 'Z', 'DOP')
METOS3D_MODEL_TRACER['NPZD-DOP'] = ('N', 'P', 'Z', 'D', 'DOP')
METOS3D_MODEL_TRACER['MITgcm-PO4-DOP'] = ('PO4', 'DOP')

TRACER_MASK = np.array(['N', 'P', 'Z', 'D', 'DOP'])

METOS3D_MODEL_TRACER_MASK = {}
METOS3D_MODEL_TRACER_MASK['N'] = np.array([True, False, False, False, False])
METOS3D_MODEL_TRACER_MASK['N-DOP'] = np.array([True, False, False, False, True])
METOS3D_MODEL_TRACER_MASK['NP-DOP'] = np.array([True, True, False, False, True])
METOS3D_MODEL_TRACER_MASK['NPZ-DOP'] = np.array([True, True, True, False, True])
METOS3D_MODEL_TRACER_MASK['NPZD-DOP'] = np.array([True, True, True, True, True])
METOS3D_MODEL_TRACER_MASK['MITgcm-PO4-DOP'] = np.array([True, False, False, False, True])

METOS3D_MODEL_TRACER_CONCENTRATIONTYP = ['constant' ,'vector']
METOS3D_MODEL_TRACER_DISTRIBUTION = ['Lognormal', 'Normal', 'OneBox', 'Uniform']
METOS3D_MODEL_TRACER_TRACERDISTRIBUTION = ['set_mass', 'random_mass']

METOS3D_MODEL_INPUT_PARAMTER_LENGTH = {}
METOS3D_MODEL_INPUT_PARAMTER_LENGTH['N'] = 5
METOS3D_MODEL_INPUT_PARAMTER_LENGTH['N-DOP'] = 7
METOS3D_MODEL_INPUT_PARAMTER_LENGTH['NP-DOP'] = 13
METOS3D_MODEL_INPUT_PARAMTER_LENGTH['NPZ-DOP'] = 16
METOS3D_MODEL_INPUT_PARAMTER_LENGTH['NPZD-DOP'] = 18
METOS3D_MODEL_INPUT_PARAMTER_LENGTH['MITgcm-PO4-DOP'] = 7

METOS3D_MODEL_OUTPUT_LENGTH = {}
METOS3D_MODEL_OUTPUT_LENGTH['N'] = len(METOS3D_MODEL_TRACER['N']) * METOS3D_VECTOR_LEN
METOS3D_MODEL_OUTPUT_LENGTH['N-DOP'] = len(METOS3D_MODEL_TRACER['N-DOP']) * METOS3D_VECTOR_LEN
METOS3D_MODEL_OUTPUT_LENGTH['NP-DOP'] = len(METOS3D_MODEL_TRACER['NP-DOP']) * METOS3D_VECTOR_LEN
METOS3D_MODEL_OUTPUT_LENGTH['NPZ-DOP'] = len(METOS3D_MODEL_TRACER['NPZ-DOP']) * METOS3D_VECTOR_LEN
METOS3D_MODEL_OUTPUT_LENGTH['NPZD-DOP'] = len(METOS3D_MODEL_TRACER['NPZD-DOP']) * METOS3D_VECTOR_LEN
METOS3D_MODEL_OUTPUT_LENGTH['MITgcm-PO4-DOP'] = len(METOS3D_MODEL_TRACER['MITgcm-PO4-DOP']) * METOS3D_VECTOR_LEN

try:
    METOS3D_PATH = os.environ['METOS3D_DIR']
except KeyError:
    METOS3D_PATH = ''
METOS3D_MODEL_PATH = system.system.METOS3D_MODEL_PATH
PATTERN_MODEL_FILENAME = 'metos3d-simpack-{}.exe'

PATTERN_OPTIONFILE = 'nesh_metos3d_options_{:s}_{:d}dt.txt'
PATTERN_TRACER_OUTPUT = '{}_output.petsc'
PATTERN_TRACER_OUTPUT_YEAR = '{:0>5d}_{}_output.petsc'
PATTERN_TRACER_INPUT = '{}_input.petsc'
PATTERN_TRACER_TRAJECTORY = 'sp{:0>4d}-ts{:0>4d}-{}_output.petsc'
PATTERN_OUTPUT_FILENAME = 'job_output.out'

INITIAL_CONCENTRATION = {}
INITIAL_CONCENTRATION['N'] = [2.17]
INITIAL_CONCENTRATION['N-DOP'] = [2.17, 0.0001]
INITIAL_CONCENTRATION['NP-DOP'] = [2.17, 0.0001, 0.0001]
INITIAL_CONCENTRATION['NPZ-DOP'] = [2.17, 0.0001, 0.0001, 0.0001]
INITIAL_CONCENTRATION['NPZD-DOP'] = [2.17, 0.0001, 0.0001, 0.0001, 0.0001]
INITIAL_CONCENTRATION['MITgcm-PO4-DOP'] = [2.17, 0.0001]

REFERENCE_PARAMETER = np.array([0.02, 0.48, 2.0, 2.0, 0.5, 0.088, 30.0, 0.75, 0.67, 0.04, 4.0, 0.03, 3.2, 0.01, 0.01, 0.05, 0.5, 0.858, 0.058, 0.0])
LOWER_BOUND = np.array([0.01, 0.24, 1.0, 1.0, 0.25, 0.044, 15.0, 0.05, 0.05, 0.02, 2.0, 0.015, 1.6, 0.005, 0.005, 0.025, 0.25, 0.7, 0.029, 0.0])
UPPER_BOUND = np.array([0.05, 0.72, 4.0, 4.0, 1.0, 0.176, 60.0, 0.95, 0.95, 0.08, 6.0, 0.045, 4.8, 0.015, 0.015, 0.1, 1.0, 1.5, 0.087, 0.0])
PARAMETER_RESTRICTION = {
    'N': np.array([True, False, True, False, True, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False]),
    'N-DOP': np.array([True, False, True, False, True, False, True, False, True, False, False, False, False, False, False, False, True, True, False, False]), 
    'NP-DOP': np.array([True, True, True, True, True, True, True, False, True, True, True, False, False, True, False, False, True, True, False, False]),
    'NPZ-DOP': np.array([True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, False, True, True, False, False]),
    'NPZD-DOP': np.array([True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True]),
    'MITgcm-PO4-DOP': np.array([True, False, True, False, True, False, True, False, True, False, False, False, False, False, False, False, True, True, False, False])
}
PARAMETER_NAMES = np.array(['k_w', 'k_c', 'mu_P', 'mu_Z', 'K_N', 'K_P', 'K_I', 'simga_Z', 'sigma_DOP', 'lambda_P', 'kappa_P', 'lambda_Z', 'kappa_Z', 'lambda_prime_P', 'lambda_prime_Z', 'lambda_prime_D', 'lambda_prime_DOP', 'b', 'a_D', 'b_D'])
PARAMETER_NAMES_LATEX = np.array(['k_w', 'k_c', '\mu_P', '\mu_Z', 'K_N', 'K_P', 'K_I', '\sigma_Z', '\sigma_{DOP}', '\lambda_P', '\kappa_P', '\lambda_Z', '\kappa_Z', '\lambda^{\prime}_P', '\lambda^{\prime}_Z', '\lambda^{\prime}_D', '\lambda^{\prime}_{DOP}', 'b', 'a_D', 'b_D'])
PARAMETER_UNITS_LATEX = np.array(['m^{-1}', '(mmol\, P\, m^{-3})^{-1} m^{-1}', 'd^{-1}', 'd^{-1}', 'mmol\, P\, m^{-3}', 'mmol\, P\, m^{-3}', 'W\, m^{-2}', '', '', 'd^{-1}', '(mmol\, P\, m^{-3})^{-1} d^{-1}', 'd^{-1}', '(mmol\, P\, m^{-3})^{-1} d^{-1}', 'd^{-1}', 'd^{-1}', 'd^{-1}', 'yr^{-1}', '', 'd^{-1}', 'm\,d^{-1}'])

