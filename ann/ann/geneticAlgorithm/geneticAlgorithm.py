#!/usr/bin/env python
# -*- coding: utf8 -*

import random
import copy

from ann.geneticAlgorithm.AbstractClassGeneticAlgorithm import AbstractClassGeneticAlgorithm
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.genome import Genome


class GeneticAlgorithm(AbstractClassGeneticAlgorithm):
    """
    Implementation of a genetic algorithm for evolving an artificial neural network.
    @author: Markus Pfeil
    """

    def __init__(self, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, populationSize = 10, retain = 0.15, random_select = 0.1, mutate_chance = 0.3):
        """
        Initialize the genetic algorithm.
        @author: Markus Pfeil
        """
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)
        assert type(populationSize) is int and 0 < populationSize
        assert type(retain) is float and 0 <= retain and retain <= 1
        assert type(random_select) is float and 0 <= random_select and random_select <= 1
        assert type(mutate_chance) is float and 0 <= mutate_chance and mutate_chance <= 1

        self._retain = retain
        self._random_select = random_select
        self._mutate_chance = mutate_chance

        AbstractClassGeneticAlgorithm.__init__(self, all_possible_genes=all_possible_genes, select_possible_genes=select_possible_genes, populationSize=populationSize)


    def _init_genome(self):
        """
        Initilialize a new genome implementing the abstract class Genome.
        @author: Markus Pfeil
        """
        return self._create_genome()


    def _create_genome(self, motherId=0, fatherId=0, geneparam={}):
        """
        Initilialize a new genome implementing the abstract class Genome.
        @author: Markus Pfeil
        """
        assert type(motherId) is int and 0 <= motherId
        assert type(fatherId) is int and 0 <= fatherId
        assert type(geneparam) is dict

        return Genome(all_possible_genes=self._all_possible_genes, select_possible_genes=self._select_possible_genes, geneparam=geneparam, u_ID=self._ids.get_next_ID(), mother_ID=motherId, father_ID=fatherId, generation=self._ids.get_Gen())


    def _breed(self, parents, child_count=1):
        """
        Make children from parental genes.
        @author: Markus Pfeil
        """
        assert type(parents) is list and len(parents) == 2
        assert type(child_count) is int and child_count > 0

        mother = parents[0]
        father = parents[1]

        children = []

        for _ in range(child_count): 
            child = {}

            #Recombination of the mother's and father's genes
            for key in self._all_possible_genes:
                child[key] = random.choice([mother.geneparam[key], father.geneparam[key]])
                
            if child['nb_layers'] != len(child['nb_neurons']):
                child['nb_layers'] = len(child['nb_neurons'])

            #Initialize a new genome
            genome = self.create_genome(motherId=mother.u_ID, fatherId=father.u_ID, geneparam=child)

            #Randomly mutate one gene
            if self._mutate_chance > random.random(): 
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
        #Increase generation 
        self._ids.increase_Gen()

        #Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in population]

        #Use those scores to fill in the master list
        for genome in population:
            self._master.set_accuracy(genome)

        #Sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]

        #Get the number we want to keep unchanged for the next cycle
        retain_length = int(len(graded) * self._retain)

        #Keep the top X percent (as defined in self.retain) without changing them
        new_generation = graded[:retain_length]

        #Randomly keep some of the lower scoring ones and mutate them
        for genome in graded[retain_length:]:
            if self._random_select > random.random():
                gtc = copy.deepcopy(genome)

                while self._master.is_duplicate(gtc):
                    gtc.mutate()

                gtc.set_generation(self._ids.get_Gen())
                new_generation.append(gtc)
                self._master.add_genome(gtc)
        
        #Two genome are mandatory as parents for the new generation
        while len(new_generation) < 2:
            genome = random.choice(graded[retain_length:])
            gtc = copy.deepcopy(genome)

            while self._master.is_duplicate(gtc):
                gtc.mutate()

            gtc.set_generation(self._ids.get_Gen())
            new_generation.append(gtc)
            self._master.add_genome(gtc)

        #How many spots we have to fill using breeding
        ng_length = len(new_generation)
        desired_length = self._populationSize - ng_length

        children = []

        #Breed genomes using pairs of remaining genomes
        while len(children) < desired_length:

            #Randomly select a distinct mother and father
            parents = random.sample(range(ng_length), k=2)

            female = new_generation[parents[0]]
            male   = new_generation[parents[1]]

            #Recombine and mutate
            babies = self._breed([female, male])

            #Add the children up to the desired_length
            for baby in babies:
                if len(children) < desired_length:
                    children.append(baby)

        new_generation.extend(children)

        return new_generation
