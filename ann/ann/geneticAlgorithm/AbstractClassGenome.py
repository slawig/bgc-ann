#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import numpy as np
import random
import logging
import hashlib

import metos3dutil.metos3d.constants as Metos3d_Constants
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.train import train_and_score

class AbstractClassGenome(ABC):
    """
    Abstract class of the genome implementation.
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    @author: Markus Pfeil
    """

    def __init__(self, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, geneparam = {}, u_ID = 0, generation = 0):
        """
        Initialize a genome.
        @author: Markus Pfeil
        """
        assert type(generation) is int and 0 <= generation
        assert type(u_ID) is int and 0 <= u_ID
        assert type(geneparam) is dict
        assert all_possible_genes is None or type(all_possible_genes) is dict
        assert select_possible_genes is None or type(select_possible_genes) is dict

        self.accuracy = 0.0
        self._trained = False
        self.all_possible_genes = all_possible_genes
        self._select_possible_genes = select_possible_genes
        self.geneparam = geneparam #(dict): represents actual genome parameters
        self.u_ID = u_ID
        self.generation = generation
        
        #Hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()


    def getTrainingStatus(self):
        """
        Get the training status of a genome.
        @author: Markus Pfeil
        """
        return self._trained


    def getUId(self):
        """
        Get the u_ID of the genome.
        @author: Markus Pfeil
        """
        return self.u_ID


    def getGeneration(self):
        """
        Get the generation of the genome.
        @author: Markus Pfeil
        """
        return self.generation


    def getMaxepoches(self):
        """
        Get the number of epoches.
        @author: Markus Pfeil
        """
        return 0 if not self.geneparam else self.geneparam['maxepoches']


    def getSETepoches(self):
        """
        Get the number of epoches for the SET algorithm.
        @author: Markus Pfeil
        """
        return 1 if not self.geneparam else self.geneparam['SETepoches']


    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        @author: Markus Pfeil
        """
        genh = ""
        for key in self.geneparam:
            genh = genh + str(self.geneparam[key])

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()
        self.accuracy = 0.0
        self._trained = False


    @abstractmethod
    def set_ancestor(self, ancestor):
        """
        Set the ancestor/ancestors of the genome.
        @author: Markus Pfeil
        """
        pass


    def set_genes_random(self):
        """
        Create a random genome.
        @author: Markus Pfeil
        """
        self.set_ancestor(0) #No parents
    
        #Set random genes    
        for key in self.all_possible_genes:
            if self._select_possible_genes[key] == 'list':
                self.geneparam[key] = random.choice(self.all_possible_genes[key])
            elif self._select_possible_genes[key] == 'int':
                self.geneparam[key] = random.randint(self.all_possible_genes[key][0], self.all_possible_genes[key][1])
            elif self._select_possible_genes[key] == 'real':
                self.geneparam[key] = random.uniform(self.all_possible_genes[key][0], self.all_possible_genes[key][1])
            else:
                assert (False), "Unknown selection strategy"
        
        #Sort neurons in ascending order
        self.geneparam['nb_neurons'] = np.sort([random.randint(self.all_possible_genes['nb_neurons'][0], self.all_possible_genes['nb_neurons'][1]) for _ in range(self.geneparam['nb_layers'])]).tolist()
        
        #SETepoches must be less than or equal maxepoches
        while self.geneparam['SETepoches'] > self.geneparam['maxepoches']:
            self.geneparam['SETepoches'] = random.randint(self.all_possible_genes['SETepoches'][0], self.all_possible_genes['SETepoches'][1])

        self.update_hash()


    @abstractmethod
    def mutate(self):
        """
        Randomly mutate genes in the genome.
        @author: Markus Pfeil
        """
        pass


    def set_generation(self, generation):
        """
        Set the generation for a genome.
        Needed when a genome is passed on from one generation to the next.
        The id stays the same, but the generation is increased.
        @author: Markus Pfeil
        """ 
        assert type(generation) is int and 0 <= generation
  
        self.generation = generation


    def train(self, metos3dModel='N', gid=0, indexMin=ANN_Constants.PARAMETERID_MAX_TEST+1, indexMax=ANN_Constants.PARAMETERID_MAX, trainingSize=None):
        """
        Train the genome and record the accuracy.
        @author: Markus Pfeil
        """
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(gid) is int and 0 <= gid
        assert type(indexMin) is int and 0 <= indexMin
        assert type(indexMax) is int and indexMin < indexMax and indexMax <= ANN_Constants.PARAMETERID_MAX
        assert trainingSize is None or type(trainingSize) is int and 0 < trainingSize and trainingSize <= indexMax - indexMin

        #Don't bother retraining ones we already trained
        if not self._trained:
            self.accuracy = train_and_score(self, metos3dModel, gid, indexMin=indexMin, indexMax=indexMax, trainingSize=trainingSize)
            self._trained = True


    def set_trained_accuracy(self, accuracy):
        """
        Set the trained accuracy.
        @author: Markus Pfeil
        """
        assert type(accuracy) is float and accuracy >= 0

        if not self._trained:
            self.accuracy = accuracy
            self._trained = True


    @abstractmethod
    def print_genome(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def print_genome_ma(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        pass


    def print_geneparam(self):
        """
        Print the list of nb_neurons as single list.
        @author: Markus Pfeil
        """
        logging.info('***Genome: {}***'.format(self.geneparam))
