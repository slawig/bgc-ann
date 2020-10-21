#!/usr/bin/env python
# -*- coding: utf8 -*

import copy
import logging
import random

import metos3dutil.metos3d.constants as Metos3d_Constants
from ann.geneticAlgorithm.AbstractClassGeneticAlgorithm import AbstractClassGeneticAlgorithm
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.genomeRechenberg import Genome


class GeneticAlgorithm(AbstractClassGeneticAlgorithm):
    """
    Implementation of the genetic algorithm of Rechenberg for evolving an artificial neural network.
    @author: Markus Pfeil
    """

    def __init__(self, gid = None, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, populationSize = 10, generations = 50, metos3dModel = 'N', alpha = 1.3, offspring = 10):
        """
        Initialize the genetic algorithm of Rechenberg.
        @author: Markus Pfeil
        """
        assert gid is None or type(gid) is int and 0 <= gid
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)
        assert type(populationSize) is int and 0 < populationSize
        assert type(generations) is int and 0 < generations
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(alpha) is float and 0 < alpha
        assert type(offspring) is int and 0 < offspring
        
        self._alpha = alpha
        self._offspring = offspring

        AbstractClassGeneticAlgorithm.__init__(self, gid=gid, all_possible_genes=all_possible_genes, select_possible_genes=select_possible_genes, populationSize=populationSize, generations=generations, metos3dModel=metos3dModel)       
 

    def _init_genome(self):
        """
        Initilialize a new genome implementing the abstract class Genome.
        @author: Markus Pfeil
        """
        return self._create_genome()


    def _create_genome(self, elterId=0, geneparam={}, mutability=random.random()):
        """
        Initilialize a new genome implementing the abstract class Genome.
        @author: Markus Pfeil
        """
        assert type(elterId) is int and 0 <= elterId
        assert type(geneparam) is dict
        assert type(mutability) is float and 0 <= mutability
 
        return Genome(all_possible_genes=self._all_possible_genes, select_possible_genes=self._select_possible_genes, geneparam=geneparam, u_ID=self._ids.get_next_ID(), elter_ID=elterId, mutability=mutability, generation=self._ids.get_Gen(), alpha=self._alpha)


    def _breed(self, elter, child_count=1):
        """
        Make childs from parental genes.
        @author: Markus Pfeil
        """
        assert type(elter) is not None
        assert type(child_count) is int and child_count > 0

        children = []

        for _ in range(child_count):
            #Copy gene for the child
            child = {}
            for key in self._all_possible_genes:
                child[key] = elter.geneparam[key]
        
            #Create new mutability for the child
            zeta = random.choice([self._alpha, 1.0 / self._alpha])
            mutability = zeta * elter.mutability
        
            #Initialize a new genome
            genome = self._create_genome(elterId=elter.getUId(), geneparam=child, mutability=mutability)

            #Randomly mutate
            genome.mutate()

            #Do we have a unique child or are we just retraining one we already have anyway?
            while self._master.is_duplicate(genome):
                genome.mutate()

            self._master.add_genome(genome)
            children.append(genome)

        assert len(children) == child_count
        return children


    def _evolve(self):
        """
        Evolve a population of genomes.
        @author: Markus Pfeil
        """
        #Selection
        #Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in self._genomes]
        
        #Use those scores to fill in the master list
        for genome in self._genomes:
            self._master.set_accuracy(genome)
        
        #Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]
        
        #Variation/mutation for the next generation
        #Increase generation 
        self._ids.increase_Gen()
        
        new_generation = copy.deepcopy(graded[:self._populationSize])
        
        for _ in range(self._offspring):
            new_generation.extend(self._breed(copy.deepcopy(random.choice(self._genomes))))
       
        self._genomes = new_generation[:self._populationSize+self._offspring]


    def _readTrainedGeneration(self):
        """
        Read the generations already trained.
        @author: Markus Pfeil
        """
        checkNextGeneration = True
        while checkNextGeneration and self._generation < self._generations + 1:
            (checkNextGeneration, genomes) = self._readGenomesGeneration()
            if not checkNextGeneration and len(genomes) == 0 and self._generation > 1:
                logging.info('***Evolve genomes for generation {:d}***'.format(self._generation))
                self._evolve()
            else:
                if self._generation == 1:
                    self._genomes = genomes
                elif self._generation == 2:
                    self._genomes = sorted(self._genomes, key=lambda x: x.accuracy) + genomes
                else:
                    self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)[:self._populationSize] + genomes

            if checkNextGeneration:
                logging.info('***Generation average: {:.5e}***'.format(self._getAverageAccuracy()))
                self._printGenomes()

                self._population[self._generation] = sorted(self._genomes, key=lambda x: x.accuracy)[:self._populationSize]

                self._updateUidDic()
                self._generation += 1

