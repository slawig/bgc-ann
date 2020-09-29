#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import ann.network.constants as ANN_Constants

dbFilename = 'ANN_Database.db'
dbPath = os.path.join(ANN_Constants.PATH, 'Database', dbFilename)


CONCENTRATIONID_CONSTANT = {'N': 0, 'N-DOP': 1, 'NP-DOP': 2, 'NPZ-DOP': 3, 'NPZD-DOP': 4, 'MITgcm-PO4-DOP': 1}