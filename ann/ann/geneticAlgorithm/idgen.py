#!/usr/bin/env python
# -*- coding: utf8 -*

import logging

class IDgen():
    """
    Generate unique genome IDs.
    @author: Markus Pfeil
    """

    def __init__(self):
        """
        Keep track of IDs.
        @author: Markus Pfeil
        """
        self._id = 0
        self._generation = 1


    def get_next_ID(self):
        """
        Increase the currentId and return the increase id
        @author: Markus Pfeil
        """
        self._id += 1
        logging.debug('***IDgen: Increase the current id to {:d}***'.format(self._id))
        return self._id


    def increase_Gen(self):
        """
        Increase the generation
        @author: Markus Pfeil
        """
        self._generation += 1
        logging.debug('***IDgen: Increase the generation to {:d}***'.format(self._generation))


    def get_Gen(self):
        """
        Return the current generation
        @author: Markus Pfeil
        """
        return self._generation
