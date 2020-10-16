#!/usr/bin/env python
# -*- coding: utf8 -*

import copy
import logging
import random

import metos3dutil.metos3d.constants as Metos3d_Constants
from ann.geneticAlgorithm.AbstractClassGeneticAlgorithm import AbstractClassGeneticAlgorithm
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.genome import Genome


class GeneticAlgorithm(AbstractClassGeneticAlgorithm):
    """
    Implementation of a genetic algorithm for evolving an artificial neural network.
    @author: Markus Pfeil
    """

    def __init__(self, gid = None, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, populationSize = 10, generations = 50, metos3dModel = 'N', retain = 0.15, random_select = 0.1, mutate_chance = 0.3):
        """
        Initialize the genetic algorithm.
        @author: Markus Pfeil
        """
        assert gid is None or type(gid) is int and 0 <= gid
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)
        assert type(populationSize) is int and 0 < populationSize
        assert type(generations) is int and 0 < generations
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS
        assert type(retain) is float and 0 <= retain and retain <= 1
        assert type(random_select) is float and 0 <= random_select and random_select <= 1
        assert type(mutate_chance) is float and 0 <= mutate_chance and mutate_chance <= 1

        self._retain = retain
        self._random_select = random_select
        self._mutate_chance = mutate_chance
        self._retain_length = int(populationSize * self._retain)
        AbstractClassGeneticAlgorithm.__init__(self, gid=gid, all_possible_genes=all_possible_genes, select_possible_genes=select_possible_genes, populationSize=populationSize, generations=generations, metos3dModel=metos3dModel)


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
            genome = self._create_genome(motherId=mother.u_ID, fatherId=father.u_ID, geneparam=child)

            #Randomly mutate one gene
            if self._mutate_chance > random.random(): 
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
        #Increase generation 
        self._ids.increase_Gen()

        #Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in self._genomes]

        #Use those scores to fill in the master list
        for genome in self._genomes:
            self._master.set_accuracy(genome)

        #Sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])]

        #Keep the top X percent (as defined in self.retain) without changing them
        new_generation = graded[:self._retain_length]

        #Randomly keep some of the lower scoring ones and mutate them
        for genome in graded[self._retain_length:]:
            if self._random_select > random.random():
                gtc = copy.deepcopy(genome)

                while self._master.is_duplicate(gtc):
                    gtc.mutate()

                gtc.set_generation(self._ids.get_Gen())
                new_generation.append(gtc)
                self._master.add_genome(gtc)
        
        #Two genome are mandatory as parents for the new generation
        while len(new_generation) < 2:
            genome = random.choice(graded[self._retain_length:])
            gtc = copy.deepcopy(genome)

            while self._master.is_duplicate(gtc):
                gtc.mutate()

            gtc.set_generation(self._ids.get_Gen())
            new_generation.append(gtc)
            self._master.add_genome(gtc)

        #Current number of genomes in the new genertion using for recombination
        ng_length = len(new_generation)

        #Breed genomes using pairs of remaining genomes
        while len(new_generation) < self._populationSize:

            #Randomly select a distinct mother and father
            parents = random.sample(range(ng_length), k=2)

            mother = new_generation[parents[0]]
            fahter = new_generation[parents[1]]

            #Recombine and mutate
            children = self._breed([mother, father])

            #Add the children up to the desired_length
            for child in children:
                if len(new_generation) < self._populationSize:
                    new_generation.append(child)

        self._genomes = new_generation


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
                self._genomes = self.evolve(self._genomes)
            else:
                if self._generation == 1:
                    self._genomes = genomes
                else:
                    self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)[:self._retain_length] + genomes

            if checkNextGeneration:
                logging.info('***Generation average: {:.5e}***'.format(self._getAverageAccuracy()))
                self._printGenomes()

                self._population[self._generation] = self._genomes
                self._updateUidList()
                self._generation += 1

