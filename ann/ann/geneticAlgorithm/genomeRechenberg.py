#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import random
import logging
import hashlib

from ann.geneticAlgorithm.AbstractClassGenome import AbstractClassGenome
import ann.geneticAlgorithm.constants as GA_Constants

class Genome(AbstractClassGenome):
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.) using the genetic algorithm of Rechenberg.
    @author: Markus Pfeil
    """

    def __init__(self, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, geneparam = {}, u_ID = 0, elter_ID = 0, mutability = 1.0, generation = 0, alpha=1.3):
        """
        Initialize a genome.
        @author: Markus Pfeil
        """
        assert type(generation) is int and 0 <= generation
        assert type(u_ID) is int and 0 <= u_ID
        assert type(elter_ID) is int and 0 <= elter_ID
        assert type(geneparam) is dict
        assert type(mutability) is float and 0 <= mutability
        assert type(alpha) is float and 0 < alpha
        assert all_possible_genes is None or type(all_possible_genes) is dict
        assert select_possible_genes is None or type(select_possible_genes) is dict

        self.elter_ID = elter_ID
        self._alpha = alpha
        self.mutability = mutability
        while self.mutability == 0.0:
            self.mutability = random.random()

        AbstractClassGenome.__init__(self, all_possible_genes=all_possible_genes, select_possible_genes=select_possible_genes, geneparam=geneparam, u_ID=u_ID, generation=generation)
        
   
    def set_ancestor(self, ancestor):
        """
        Set the ancestor/ancestors of the genome.
        @author: Markus Pfeil
        """
        assert type(ancestor) is int and 0 <= ancestor

        self.elter_ID = ancestor

 
    def set_mutability(self, mutability):
        """
        Set the mutability.
        @author: Markus Pfeil
        """
        self.mutability = mutability


    def mutate(self):
        """
        Randomly mutate the genes in the genome.
        Calculate the standard deviation with the 6 sigma rule
        @author: Markus Pfeil
        """
        for key in self.all_possible_genes:
            if key == 'nb_neurons':
                for i in range(0, len(self.geneparam[key])):
                    z = self.mutability * np.random.normal(0.0, (self.all_possible_genes[key][1] - self.all_possible_genes[key][0]) / 6.0)
                    self.geneparam[key][i] = int(round(self.geneparam[key][i] + z))
   
                    # Set the value to the upper/lower bound if the value is too high/small
                    if self.geneparam[key][i] < self.all_possible_genes[key][0]:
                        self.geneparam[key][i] = self.all_possible_genes[key][0]
                    elif self.geneparam[key][i] > self.all_possible_genes[key][1]:
                        self.geneparam[key][i] = self.all_possible_genes[key][1]
            elif self._select_possible_genes[key] == 'list':
                z = self.mutability * np.abs(np.random.normal(0.0, 1.0/len(self.all_possible_genes[key])))
                if z > 1.0/len(self.all_possible_genes[key]) and len(self.all_possible_genes[key]) > 1:
                    select_geneparam = random.choice(self.all_possible_genes[key])
                    while select_geneparam == self.geneparam[key]:
                        select_geneparam = random.choice(self.all_possible_genes[key])
                    self.geneparam[key] = select_geneparam
            elif self._select_possible_genes[key] in ['int', 'real']:
                z = self.mutability * np.random.normal(0.0, (self.all_possible_genes[key][1] - self.all_possible_genes[key][0]) / 6.0)
                self.geneparam[key] = self.geneparam[key] + z
                
                if self._select_possible_genes[key] == 'int':
                    self.geneparam[key] = int(round(self.geneparam[key]))
                    
                # Set the value to the upper/lower bound if the value is too high/small
                if self.geneparam[key] < self.all_possible_genes[key][0]:
                    self.geneparam[key] = self.all_possible_genes[key][0]
                elif self.geneparam[key] > self.all_possible_genes[key][1]:
                    self.geneparam[key] = self.all_possible_genes[key][1]
            else:
                assert (False), "Unknown selection strategy"
                
        while self.geneparam['nb_layers'] != len(self.geneparam['nb_neurons']):
            if self.geneparam['nb_layers'] < len(self.geneparam['nb_neurons']):
                self.geneparam['nb_neurons'] = np.sort(random.sample(self.geneparam['nb_neurons'], self.geneparam['nb_layers'])).tolist()
            else:
                self.geneparam['nb_neurons'].append(random.randint(self.all_possible_genes['nb_neurons'][0], self.all_possible_genes['nb_neurons'][1]))
                self.geneparam['nb_neurons'] = np.sort(self.geneparam['nb_neurons']).tolist()
                
        while self.geneparam['SETepoches'] > self.geneparam['maxepoches']:
            z = self.mutability * np.random.normal(0.0, (self.all_possible_genes['SETepoches'][1] - self.all_possible_genes['SETepoches'][0]) / 6.0)
            self.geneparam['SETepoches'] = self.geneparam['SETepoches'] + z
            self.geneparam['SETepoches'] = int(round(self.geneparam['SETepoches']))
                
            # Set the value to the upper/lower bound if the value is too high/small
            if self.geneparam['SETepoches'] < self.all_possible_genes['SETepoches'][0]:
                self.geneparam['SETepoches'] = self.all_possible_genes['SETepoches'][0]
            elif self.geneparam['SETepoches'] > self.all_possible_genes['SETepoches'][1]:
                self.geneparam['SETepoches'] = self.all_possible_genes['SETepoches'][1]
        
        self.update_hash()


    def print_genome(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        self.print_geneparam()
        logging.info('Genome: Generation: {:d}'.format(self.generation))
        logging.info('Genome: UniID: {:d}'.format(self.u_ID))
        logging.info('Genome: Elter: {:d}'.format(self.elter_ID))
        logging.info('Genome: Mutability: {:.2f}'.format(self.mutability))
        logging.info('Genome: Alpha: {.2f}'.format(self._alpha))
        logging.info('Genome: Accuracy: {:.5e}'.format(self.accuracy))
        logging.info('Genome: Hash: {:s}'.format(self.hash))


    def print_genome_ma(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        self.print_geneparam()
        logging.info('Genome: Generation: {:d} UniID: {:d} Elter: {:d} Mutability: {:.2f}  Alpha: {:.2f} Accuracy: {.5e} Hash: {:s}'.format(self.generation, self.u_ID, self.elter_ID, self.mutability, self._alpha, self.accuracy, self.hash))
