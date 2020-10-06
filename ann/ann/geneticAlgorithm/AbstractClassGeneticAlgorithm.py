#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
from functools import reduce
from operator import add

import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.AbstractClassGenome import AbstractClassGenome
from ann.geneticAlgorithm.genome import Genome
from ann.geneticAlgorithm.idgen import IDgen
from ann.geneticAlgorithm.allGenomes import AllGenomes


class AbstractClassGeneticAlgorithm(ABC):
    """
    Abstract class for the implementation of a genetic algorithm.
    @author: Markus Pfeil
    """

    def __init__(self, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, populationSize = 10, generations = 50):
        """
        Initialize a genetic algorithm.
        @author: Markus Pfeil
        """
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)
        assert type(populationSize) is int and 0 < populationSize
        assert type(generations) is int and 0 < generations

        self._all_possible_genes = all_possible_genes
        self._select_possible_genes = select_possible_genes
        self._populationSize = populationSize
        self._generations = generations

        self._population = {}
        self._genomesUidDic = {}

        self._ids = IDgen()
        self._master = AllGenomes()	# This is where we will store all genomes


    def create_population(self, count):
        """
        Create a population of with random genomes.
        @author: Markus Pfeil
        """
        self._populationSize = count
        population = []
        for i in range(0, self._populationSize):
            # Initialize a new genome
            genome = self._init_genome()

            # Set it to random parameters.
            genome.set_genes_random()

            # Make sure that the genome is unique
            while self._master.is_duplicate(genome):
                genome.mutate()

            # Add the genome to the population and to the master list
            population.append(genome)
            self._master.add_genome(genome)

        return population


    @abstractmethod
    def _init_genome(self):
        """
        Initilialize a new genome implementing the abstract class Genome.
        @author: Markus Pfeil
        """
        pass


    def add_genome(self, genome, increaseId=False):
        """
        Add the genome to the population.
        @author: Markus Pfeil
        """
        self._master.add_genome(genome)

        if increaseId:
            self._ids.get_next_ID()


    @staticmethod
    def fitness(genome):
        """
        Return the accuracy, which is the fitness function.
        @author: Markus Pfeil
        """
        return genome.accuracy


    def grade(self, population):
        """
        Find average fitness for a population.
        @author: Markus Pfeil
        """
        summed = reduce(add, (self.fitness(genome) for genome in population))
        return summed / float((len(population)))


    @abstractmethod
    def _breed(self, parents, child_count=1):
        """
        Make childs from the parental genes.
        @author: Markus Pfeil
        """
        pass


    @abstractmethod
    def evolve(self, population):
        """
        Evolve a population of genomes.
        @author: Markus Pfeil
        """
        pass


    def run(self):
        """
        Run the genetic algorithm for the given number of generations.
        @author: Markus Pfeil
        """
        logging.info('***Run the genetic algorithm with {:d} generations and population size {:d}***'.format(self._generations, self._populationSize))

        genomes = []
        #Try to read the genomes for the first generations from the file system
        generation = 0
        checkGeneration = True
        
        while generation < self._generations and checkGeneration:
            logging.debug('***Read genomes for the generation {:d}***'.format(generation))
            #Read the trained genomes of the given generation from the file system
            (checkGeneration, genomesGeneration) = readGeneration()

            if len(genomesGeneration) > 0:
                self._insertPopulation(generation, genomesGeneration)
                self._updateUidDic(generation, genomesGeneration)
                genomes.extend(genomesGeneration)
                genomes = sorted(genomes, key=lambda x: x.accuracy)
                genomes = genomes[0:self._populationSize]
                generation += 1
                if generation > 1:
                    self._ids.increase_Gen()
        
        #Decrease generation because the index starts with 0
        generation -= 1

        #Create population only for the first generation
        if generation == -1 and len(genomes) == 0:
            generation = 0
            genomes = self.create_population()

        #Evolve the generation
        #TODO


    def _insertPopulation(self, generation, genomes):
        """
        Insert the genomes for the given generation into the population dictionary.
        @author: Markus Pfeil
        """
        assert type(generation) is int and 0 <= generation
        assert type(genomes) is list and len(genomes) > 0

        #Insert the genomes to the whole population
        assert not generation in self._population
        for genome in genomes:
            assert generation == genome.getGeneration()

        self._population[generation] = sorted(genomes, key=lambda x: x.getUId())


    def _updateUidDic(self, generation, genomes):
        """
        Update the list of uids in the genomesUidDic.
        @author: Markus Pfeil
        """
        assert type(generation) is int and 0 <= generation
        assert type(genomes) is list and len(genomes) > 0

        for genome in genomes:
            assert generation == genome.getGeneration()
            uid = genome.getUId()

            #Test if the uid is already present in the uidDic
            uidKey = []
            for key in self._genomesUidDic:
                if uid in self._genomesUidDic[key]:
                    uidKey.append(key)

            if not generation in uidKey:
                try:
                    if not generation in self._genomesUidDic:
                        self._genomesUidDic[generation] = [uid]
                    else:
                        self._genomesUidDic[generation].append(uid)
                        self._genomesUidDic[generation] = sorted(self._genomesUidDic[generation])
                except KeyError:
                    self._genomesUidDic[generation] = [uid]


    def _containUid(self, uid):
        """
        Check if the given dictionary uidDic contains the given uid as value.
        @author: Markus Pfeil
        """
        assert type(uid) is int and 0 <= uid

        checkUid = False
        for key in self._genomesUidDic:
            if uid in self._genomesUidDic[key]:
                checkUid = True
                break

        return checkUid
