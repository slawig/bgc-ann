#!/usr/bin/env python
# -*- coding: utf8 -*

import random
import logging


class AllGenomes():
    """
    Class that stores all genomes and keeps track of all genomes trained so far, and their scores.
    Among other things, ensures that genomes are unique.
    The genome must has the varibles accuarcy and hash and implemet the function print_genome_ma().
    @author: Markus Pfeil
    """

    def __init__(self):
        """
        Initialize the class to store all genomes.
        @author: Markus Pfeil
        """
        self._population = []


    def is_duplicate(self, genome):
        """
        Exists the genome already in the population.
        For this purpose, we use the hash value of the genome.
        @author: Markus Pfeil
        """
        existsGenome = False
        for i in range(0, len(self._population)):
            if (genome.hash == self._population[i].hash):
                existsGenome = True
                break
        return existsGenome


    def add_genome(self, genome):
        """
        Add the genome to our population.
        @author: Markus Pfeil
        """
        if self.is_duplicate(genome):
            logging.info("AllGenomes: add_genome() ERROR: hash clash - duplicate genome")
            return False
        else:
            self._population.append(genome)
            return True


    def set_accuracy(self, genome):
        """
        Set the accuracy of the genome in the population.
        @author: Markus Pfeil
        """
        if self.is_duplicate(genome):
            for i in range(0,len(self._population)):
                if (genome.hash == self._population[i].hash):
                    self._population[i].accuracy = genome.accuracy
                    break
        else:
            logging.info("AllGenomes: set_accuracy() ERROR: Genome not found")


    def print_all_genomes(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        for genome in self._population:
            genome.print_genome_ma()
