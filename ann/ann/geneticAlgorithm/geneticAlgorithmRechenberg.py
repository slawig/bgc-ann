#!/usr/bin/env python
# -*- coding: utf8 -*

import random
import copy

from ann.geneticAlgorithm.AbstractClassGeneticAlgorithm import AbstractClassGeneticAlgorithm
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.genomeRechenberg import Genome


class GeneticAlgorithm(AbstractClassGeneticAlgorithm):
    """
    Implementation of the genetic algorithm of Rechenberg for evolving an artificial neural network.
    @author: Markus Pfeil
    """

    def __init__(self, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, populationSize = 10, alpha = 1.3, offspring = 10):
        """
        Initialize the genetic algorithm of Rechenberg.
        @author: Markus Pfeil
        """
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)
        assert type(populationSize) is int and 0 < populationSize
        assert type(alpha) is float and 0 < alpha
        assert type(offspring) is int and 0 < offspring
        
        self._alpha = alpha
        self._offspring = offspring

        AbstractClassGeneticAlgorithm.__init__(self, all_possible_genes=all_possible_genes, select_possible_genes=select_possible_genes, populationSize=populationSize)       
 

    def _init_genome(self):
        """
        Initilialize a new genome implementing the abstract class Genome.
        @author: Markus Pfeil
        """
        self._create_genome()


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
        assert type(elter) is list and len(elter) == 1
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
            genome = self._create_genome(elterId=elter.u_ID, geneparam=child, mutability=mutability)

            #Randomly mutate
            genome.mutate()

            #Do we have a unique child or are we just retraining one we already have anyway?
            while self._master.is_duplicate(genome):
                genome.mutate()

            self._master.add_genome(genome)
            children.append(genome)

        assert len(childen) == child_count
        return children


    def evolve(self, population):
        """
        Evolve a population of genomes.
        @author: Markus Pfeil
        """
        #Selection
        #Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in population]
        
        #Use those scores to fill in the master list
        for genome in population:
            self._master.set_accuracy(genome)
        
        #Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]
        
        #Variation/mutation for the next generation
        #Increase generation 
        self._ids.increase_Gen()
        
        new_generation = copy.deepcopy(graded[:self._populationSize])
        
        for _ in range(self._offspring):
            new_generation.extend(self._breed(random.choice(population)))
        
        return new_generation[:self._populationSize+self._offspring]
