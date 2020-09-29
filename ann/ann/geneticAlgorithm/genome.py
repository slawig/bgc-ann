#!/usr/bin/env python
# -*- coding: utf8 -*

import numpy as np
import random
import logging

from ann.geneticAlgorithm.AbstractClassGenome import AbstractClassGenome
import ann.geneticAlgorithm.constants as GA_Constants


class Genome(AbstractClassGenome):
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    @author: Markus Pfeil
    """

    def __init__(self, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, geneparam = {}, u_ID = 0, mother_ID = 0, father_ID = 0, generation = 0):
        """
        Initialize a genome.
        @author: Markus Pfeil
        """
        assert type(generation) is int and 0 <= generation
        assert type(u_ID) is int and 0 <= u_ID
        assert type(mother_ID) is int and 0 <= mother_ID
        assert type(father_ID) is int and 0 <= father_ID
        assert type(geneparam) is dict
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)

        self.parents = [mother_ID, father_ID]
        
        AbstractClassGenome.__init__(self, all_possible_genes=all_possible_genes, select_possible_genes=select_possible_genes, geneparam=geneparam, u_ID=u_ID, generation=generation)


    def set_ancestor(self, ancestor):
        """
        Set the ancestor/ancestors of the genome.
        @author: Markus Pfeil
        """
        assert type(ancestor) is int and 0 <= ancestor

        self.parents = [ancestor, ancestor]


    def mutate(self):
        """
        Randomly mutate one gene in the genome.
        @author: Markus Pfeil
        """
        #Select random a gene to mutate
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))
        current_value = self.geneparam[gene_to_mutate]
 
        while(self.geneparam[gene_to_mutate] == current_value):
            new_gene = False

            if self._select_possible_genes[gene_to_mutate] == 'list' and len(self.all_possible_genes[gene_to_mutate]) == 1:
                new_gene = True
            elif self._select_possible_genes[gene_to_mutate] in ['int', 'real'] and self.all_possible_genes[gene_to_mutate][0] == self.all_possible_genes[gene_to_mutate][1]:
                new_gene = True
            elif self._select_possible_genes[gene_to_mutate] == 'list':
                self.geneparam[gene_to_mutate] = random.choice(self.all_possible_genes[gene_to_mutate])
            elif self._select_possible_genes[gene_to_mutate] == 'int':
                self.geneparam[gene_to_mutate] = random.randint(self.all_possible_genes[gene_to_mutate][0], self.all_possible_genes[gene_to_mutate][1])
            elif self._select_possible_genes[gene_to_mutate] == 'real':
                self.geneparam[gene_to_mutate] = random.uniform(self.all_possible_genes[gene_to_mutate][0], self.all_possible_genes[gene_to_mutate][1])
            else:
                assert (False), "Unknown selection strategy"
 
            if new_gene:
                gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))
            elif gene_to_mutate == 'nb_neurons':
                self.geneparam['nb_neurons'] = np.sort([random.randint(self.all_possible_genes['nb_neurons'][0], self.all_possible_genes['nb_neurons'][1]) for _ in range(self.geneparam['nb_layers'])]).tolist()
            elif gene_to_mutate == 'nb_layers' and self.geneparam['nb_layers'] != len(self.geneparam['nb_neurons']):
                while self.geneparam['nb_layers'] != len(self.geneparam['nb_neurons']):
                    if self.geneparam['nb_layers'] < len(self.geneparam['nb_neurons']):
                        self.geneparam['nb_neurons'] = np.sort(random.sample(self.geneparam['nb_neurons'], self.geneparam['nb_layers'])).tolist()
                    else:
                        self.geneparam['nb_neurons'].append(random.randint(self.all_possible_genes['nb_neurons'][0], self.all_possible_genes['nb_neurons'][1]))
                        self.geneparam['nb_neurons'] = np.sort(self.geneparam['nb_neurons']).tolist()                
            elif gene_to_mutate in ['SETepoches', 'maxepoches']:
                while self.geneparam['SETepoches'] > self.geneparam['maxepoches']:
                    self.geneparam['SETepoches'] = random.randint(self.all_possible_genes['SETepoches'][0], self.all_possible_genes['SETepoches'][1])  

        self.update_hash()


    def set_genes_to(self, geneparam, mother_ID, father_ID):
        """
        Set genome properties.
        @author: Markus Pfeil
        """
        assert type(mother_ID) is int and 0 <= mother_ID
        assert type(father_ID) is int and 0 <= father_ID
        assert type(geneparm) is dict

        self.parents  = [mother_ID, father_ID]
        self.geneparam = geneparam

        self.update_hash()


    def print_genome(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        logging.info('Genome: Generation: {:d}'.format(self.generation))
        logging.info('Genome: UniID: {:d}'.format(self.u_ID))
        logging.info('Genome: Mother and Father: {:d}/{:d}'.format(self.parents[0], self.parents[1]))
        self.print_geneparam()
        logging.info('Genome: Accuracy: {:.5e}'.format(self.accuracy))
        logging.info('Genome: Hash: {:s}\n'.format(self.hash))


    def print_genome_ma(self):
        """
        Print out a genome.
        @author: Markus Pfeil
        """
        self.print_geneparam()
        logging.info('***Genome: Generation: {:d} UniID: {:d} Mother/Father: {:d}/{:d} Accuarcy: {.5e} Hash: {:s}***'.format(self.generation, self.u_ID, self.parents[0], self.parents[1], self.accuracy, self.hash))
