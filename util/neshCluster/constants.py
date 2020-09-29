#!/usr/bin/env python
# -*- coding: utf8 -*

# Stettings for the nesh linux cluster of the CAU Kiel
CPUNUM = {'clmedium': 32, 'cllong': 32, 'clbigmem': 32, 'clexpress': 32}
ELAPSTIM = {'clmedium': 48, 'cllong': 100, 'clbigmem': 200, 'clexpress': 2}
QUEUE = [key for key in CPUNUM]

