#!/usr/bin/env python
# -*- coding: utf8 -*

SBO_Config_Default = {'Options': None}


#SBO_Config includes special configurations used for the surrogate based optimization.
#SBO_Config defines only the differences to the default configuraiton SBO_Config_Default.
SBO_Config = {}

SBO_Config[224] = {'Options': {'maxiter': 100, 'disp': True, 'gtol': 10**(-8)}}
SBO_Config[225] = {'Options': {'maxiter': 100, 'disp': True, 'gtol': 10**(-8)}}
SBO_Config[226] = {'Options': {'maxiter': 100, 'disp': True, 'eps': 10**(-3)}}
SBO_Config[227] = {'Options': {'maxiter': 100, 'disp': True, 'eps': 10**(-3)}}
SBO_Config[228] = {'Options': {'maxiter': 100, 'disp': True, 'gtol': 10**(-8), 'eps': 10**(-3)}}
SBO_Config[229] = {'Options': {'maxiter': 100, 'disp': True, 'gtol': 10**(-8), 'eps': 10**(-3)}}

