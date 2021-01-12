#!/usr/bin/env python
# -*- coding: utf8 -*

from abc import ABC, abstractmethod
import bz2
import copy
from functools import reduce
import logging
from operator import add
import os
import pickle
import re
import shutil
import tarfile

from system.system import SYSTEM, PYTHON_PATH
if SYSTEM == 'PC':
    from standaloneComputer.JobAdministration import JobAdministration
else:
    from neshCluster.JobAdministration import JobAdministration
import metos3dutil.metos3d.constants as Metos3d_Constants
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
from ann.geneticAlgorithm.AbstractClassGenome import AbstractClassGenome
from ann.geneticAlgorithm.genome import Genome
from ann.geneticAlgorithm.idgen import IDgen
from ann.geneticAlgorithm.allGenomes import AllGenomes


class AbstractClassGeneticAlgorithm(ABC, JobAdministration):
    """
    Abstract class for the implementation of a genetic algorithm.
    @author: Markus Pfeil
    """

    def __init__(self, gid = None, all_possible_genes = GA_Constants.all_possible_genes, select_possible_genes = GA_Constants.select_possible_genes, populationSize = 10, generations = 50, metos3dModel='N'):
        """
        Initialize a genetic algorithm.
        @author: Markus Pfeil
        """
        assert gid is None or type(gid) is int and 0 <= gid
        assert type(all_possible_genes) is dict
        assert type(select_possible_genes) is dict
        assert len(all_possible_genes) == len(select_possible_genes)
        assert type(populationSize) is int and 0 < populationSize
        assert type(generations) is int and 0 < generations
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        JobAdministration.__init__(self)

        if gid is None:
            self._generateGid()
        else:
            self._pathGA = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid))
            assert os.path.exists(self._pathGA) and os.path.isdir(self._pathGA)
            self._gid = gid

        self._all_possible_genes = all_possible_genes
        self._select_possible_genes = select_possible_genes
        self._populationSize = populationSize
        self._generations = generations
        self._generation = 1
        self._metos3dModel = metos3dModel

        self._genomes = []
        self._population = {}
        self._genomesUidDic = {}

        self._ids = IDgen()
        self._master = AllGenomes()	# This is where we will store all genomes


    def _generateGid(self):
        """
        Generate a unique gid and create directory.
        @author: Markus Pfeil
        """
        self._gid = 0
        self._pathGA = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))

        while(os.path.exists(self._pathGA) and os.path.isdir(self._pathGA)):
            self._gid += 1
            self._pathGA = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))
        os.makedirs(self._pathGA, exist_ok=False)


    def _create_population(self):
        """
        Create a population of with random genomes.
        @author: Markus Pfeil
        """
        while (len(self._genomes) < self._populationSize):
            # Initialize a new genome
            genome = self._init_genome()

            # Set it to random parameters.
            genome.set_genes_random()

            # Make sure that the genome is unique
            while self._master.is_duplicate(genome):
                genome.mutate()

            # Add the genome to the population and to the master list
            self._genomes.append(copy.deepcopy(genome))
            self._master.add_genome(genome)


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
    def _evolve(self):
        """
        Evolve a population of genomes.
        @author: Markus Pfeil
        """
        pass


    def run(self, config={}):
        """
        Run the training of an ANN using the genetic algorithm.
        @author: Markus Pfeil
        """
        assert type(config) is dict

        logging.info('***Run the genetic algorithm with gid {:d} for {:d} generations with a population size of {:d}***'.format(self._gid, self._generations, self._populationSize))

        self._initGeneticAlgorithm()
        self._runEvolution(config=config)


    def _initGeneticAlgorithm(self):
        """
        Initialisation of the training process using the genetic algorithm.
        @author: Markus Pfeil
        """
        #Read the already trained genomes
        self._readTrainedGeneration()

        #Create initial population
        if self._generation == 1 and len(self._genomes) == 0:
            self._create_population()


    def _runEvolution(self, config={}):
        """
        Evolve the popultion over the given number of generations.
        @author: Markus Pfeil
        """
        assert type(config) is dict

        while self._generation <= self._generations:
            logging.info('***Now in generation {:d} of {:d}***'.format(self._generation, self._generations))

            self._updateUidDic()
            logging.debug('***List of uids in this generation: {}***'.format(self._genomesUidDic[self._generation]))

            #Train and get accuracy for networks/genomes
            self._trainGenomes(config=config)

            #Get the average accuracy for this generation and print the genomes
            self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)
            self._population[self._generation] = copy.deepcopy(self._genomes[:self._populationSize])
            logging.info('***Generation average: {:.5e}***'.format(self._getAverageAccuracy()))
            self._printGenomes()

            #Evolve, except on the last iteration
            if self._generation < self._generations:
                logging.info('***Evolve genomes for generation {:d}***'.format(self._generation + 1))
                self._evolve()

            #Create data backup for the generation
            logging.info('***Generate backup of the generation {:d}***'.format(self._generation))
            self.generateBackup(generation=self._generation)

            self._generation += 1

        #Sort our final population according to performance
        self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)

        #Print out the networks/genomes
        logging.info('***Sorted list of networks/genomes***')
        self._printGenomes()

        #Save file with the id and generation of the best ANN
        self._generateGenomeFile(self._genomes[0])

        #Create data backup for the whole calculation
        self.generateBackup()


    @abstractmethod
    def _readTrainedGeneration(self):
        """
        Read the generations already trained.
        @author: Markus Pfeil
        """
        pass


    def _readGenomesGeneration(self):
        """
        Read the genomes for one generation.
        Check, if all genomes for this generation are already trained.
        @author: Markus Pfeil
        """
        logging.debug('***Check generation {:d} for gid {:d} and population size {:d}***'.format(self._generation, self._gid, self._populationSize))

        checkGeneration = True
        genomes = []

        #Read the genomes in the directories for the given generation
        genomesDirectories = list(filter(lambda x: os.path.isdir(os.path.join(self._pathGA, x)) and x.startswith(GA_Constants.GENETIC_ALGORITHM_MODEL.format(self._generation, 0).rsplit('_', maxsplit=1)[0]), os.listdir(os.path.join(self._pathGA))))

        if len(genomesDirectories) == 0:
            checkGeneration = False

        for dic in genomesDirectories:
            genomeUid = int(dic.rsplit('_', maxsplit=1)[1])

            if self._checkAnn(genomeUid, self._generation):
                #Training of the genomes is already finished
                genomeFile = os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(self._generation, genomeUid), GA_Constants.GENOME_FILENAME.format(self._gid, self._generation, genomeUid))
                if os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                    with open(genomeFile, 'rb') as f:
                        genome = pickle.load(f)
                        accuracy = self._readGenomeAccuracy(genome)
                        genome.set_trained_accuracy(accuracy)
                        increaseId = not self._containUid(genomeUid)
                        self.add_genome(genome, increaseId=increaseId)
                        genomes.append(genome)

                #Remove jobfile and second genome file
                jobfile = os.path.join(self._pathGA, GA_Constants.PATTERN_JOBFILE_TRANING_GENOME.format(self._gid, self._generation, genomeUid))
                genomeFile = os.path.join(self._pathGA, GA_Constants.GENOME_FILENAME.format(self._gid, self._generation, genomeUid))
                if os.path.exists(jobfile) and os.path.isfile(jobfile):
                    os.remove(jobfile)
                if os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                    os.remove(genomeFile)
            else:
                #Genome has to be trained so read the original genome file
                checkGeneration = False
                genomeFile = os.path.join(self._pathGA, GA_Constants.GENOME_FILENAME.format(self._gid, self._generation, genomeUid))
                if os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                    with open(genomeFile, 'rb') as f:
                        genome = pickle.load(f)
                        increaseId = not self._containUid(genomeUid)
                        self.add_genome(genome, increaseId=increaseId)
                        genomes.append(genome)

                if os.path.exists(os.path.join(self._pathGA, dic)) and os.path.isdir(os.path.join(self._pathGA, dic)):
                    shutil.rmtree(os.path.join(self._pathGA, dic))

        #Read the genomes without directory (so far, the training of the neural network using this genome has not been started)
        genomesFiles = list(filter(lambda x: os.path.isfile(os.path.join(self._pathGA, x)) and x.startswith(GA_Constants.GENOME_FILENAME.format(self._gid, self._generation, 0).rsplit('_', maxsplit=2)[0]), os.listdir(self._pathGA)))

        for genomeFile in genomesFiles:
            matchUid = re.search(GA_Constants.RE_PATTERN_GENOME_FILENAME, genomeFile)
            if matchUid:
                genomeUid = int(matchUid.groups()[2])
            else:
                assert False

            genomeDir = os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(self._generation, genomeUid))
            genomeFile = os.path.join(self._pathGA, GA_Constants.GENOME_FILENAME.format(self._gid, self._generation, genomeUid))
            if not (os.path.exists(genomeDir) and os.path.isdir(genomeDir)) and os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                checkGeneration = False
                with open(genomeFile, 'rb') as f:
                    genome = pickle.load(f)
                    increaseId = not self._containUid(genomeUid)
                    self.add_genome(genome, increaseId=increaseId)
                    genomes.append(genome)

        return (checkGeneration, genomes)


    def _checkAnn(self, genomeUid, generation):
        """
        Check, if the directory includes the three files for the ann and the file for the gnome.
        @author: Markus Pfeil
        """
        assert type(genomeUid) is int and 0 <= genomeUid
        assert type(generation) is int and 0 < generation

        logging.info('***Check ANN (gid: {:d}, generation: {:d}, genomeUid: {:d})***'.format(self._gid, generation, genomeUid))
        path = os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(generation, genomeUid))
        check_path = os.path.exists(path) and os.path.isdir(path)
        check_architecture = os.path.exists(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(genomeUid))) and os.path.isfile(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(genomeUid)))
        check_training = os.path.exists(os.path.join(path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(genomeUid))) and os.path.isfile(os.path.join(path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(genomeUid)))
        check_weights = os.path.exists(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(genomeUid))) and os.path.isfile(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(genomeUid)))

        check_genome = os.path.exists(os.path.join(path, GA_Constants.GENOME_FILENAME.format(self._gid, generation, genomeUid))) and os.path.isfile(os.path.join(path, GA_Constants.GENOME_FILENAME.format(self._gid, generation, genomeUid)))

        logging.debug('***Check ANN (gid: {:d}, generation: {:d}, genomeUid: {:d})***\nCheck path: {}\nCheck architecture: {}\nCheck training: {}\nCheck weights: {}\nCheck genome file: {}'.format(self._gid, generation, genomeUid, check_path, check_architecture, check_training, check_weights, check_genome))
        return check_path and check_architecture and check_training and check_weights and check_genome


    def _readGenomeAccuracy(self, genome):
        """
        Read the loss of the training from the training monitor file.
        @author: Markus Pfeil
        """
        filename = os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.getGeneration(), genome.getUId()), ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(genome.getUId()))
        assert os.path.exists(filename) and os.path.isfile(filename)
        with open(filename, 'r') as f:
            for line in f.readlines():
                matches = re.search(r'^\s*(\d+): (\d+.\d+e[+-]\d+), (\d+.\d+e[+-]\d+), (\d+.\d+e[+-]\d+), (\d+.\d+e[+-]\d+), (\d+.\d+e[+-]\d+), (\d+.\d+e[+-]\d+)', line)
                if matches:
                    epoch, loss = matches.groups()[:2]

        return float(loss)


    def _getAverageAccuracy(self):
        """
        Calculate the average accuracy for a group of networks/genomes.
        @author: Markus Pfeil
        """
        totalAccuracy = 0

        for genome in self._genomes:
            totalAccuracy += genome.accuracy

        return totalAccuracy / len(self._genomes)


    def _removeDirectory(self, genomeUid, generation):
        """
        Remove the directory of the ANN.
        @author: Markus Pfeil
        """
        assert type(genomeUid) is int and 0 <= genomeUid
        assert type(generation) is int and 0 < generation

        path = os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(generation, genomeUid))
        logging.info('***Remove directory for the ANN in generation {:d} with id {:d}: {}'.format(generation, genomeUid, path))

        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


    def saveGenome(self, genomeFilename, genome):
        """
        Save genome in a file.
        @author: Markus Pfeil
        """
        assert not os.path.exists(genomeFilename)

        with open(genomeFilename, 'wb') as f:
            pickle.dump(genome, f, pickle.HIGHEST_PROTOCOL)


    def _printGenomes(self):
        """
        Print the list of genomes.
        @author: Markus Pfeil
        """
        logging.info('-'*80)

        for genome in self._genomes:
            genome.print_genome()

        logging.info('-'*80)


    def _updateUidDic(self):
        """
        Update the list of uids in the genomesUidDic using the list of genomes of the current population.
        @author: Markus Pfeil
        """
        for genome in self._genomes:
            generation = genome.getGeneration()
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


    def _trainGenomes(self, config={}):
        """
        Train each genomes in the genomes list.
        @author: Markus Pfeil
        """
        assert type(config) is dict

        logging.info('***Train networks***')

        #Set job list of untrained genomes
        for genome in self._genomes:
            if not genome.getTrainingStatus():
                genomeFilename = GA_Constants.GENOME_FILENAME.format(self._gid, genome.getGeneration(), genome.getUId())
                if not (os.path.exists(os.path.join(self._pathGA, genomeFilename)) and os.path.isfile(os.path.join(self._pathGA, genomeFilename))):
                    self.saveGenome(os.path.join(self._pathGA, genomeFilename), genome)

                #Set queue for the training
                if genome.getMaxepoches() < 4000 or "trainingSize" in config and config["trainingSize"] <= 500:
                    queue='clmedium'
                else:
                    queue='cllong'

                # Create directory for the joboutput
                pathLogs = os.path.join(self._pathGA, 'Logs', 'Joboutput')
                os.makedirs(pathLogs, exist_ok=True)

                #Programm
                programm = 'ANN_TrainGenome.py {} {:s} {:d}'.format(genomeFilename, self._metos3dModel, self._gid)
                #Optional Parameter
                if 'indexMin' in config:
                    programm = programm + ' -indexMin {:d}'.format(config["indexMin"])
                if 'indexMax' in config:
                    programm = programm + ' -indexMax {:d}'.format(config["indexMax"])
                if 'trainingSize' in config:
                    programm = programm + ' -trainingSize {:d}'.format(config["trainingSize"])

                jobDict = {}
                jobDict['path'] = self._pathGA
                jobDict['jobFilename'] = os.path.join(self._pathGA, GA_Constants.PATTERN_JOBFILE_TRANING_GENOME.format(self._gid, genome.getGeneration(), genome.getUId()))
                jobDict['jobname'] = 'GA_{}_{}_TrainingGenome'.format(self._gid, genome.getUId())
                jobDict['joboutput'] = os.path.join(pathLogs, GA_Constants.PATTERN_JOBOUTPUT_TRANING_GENOME.format(self._gid, genome.getGeneration(), genome.getUId()))
                jobDict['programm'] = os.path.join(PYTHON_PATH, 'ann', 'ann', 'geneticAlgorithm', programm)
                jobDict['queue'] = queue
                jobDict['memory'] = 30
                jobDict['cores'] = 1
                jobDict['genomeFilename'] = genomeFilename
                jobDict['genomeUid'] = genome.getUId()
                jobDict['generation'] = genome.getGeneration()

                self.addJob(jobDict)

        #Run the training for each untrained genome
        self.runJobs()

        # Set accuracy for every genome
        for genome in self._genomes:
            accuracy = self._readGenomeAccuracy(genome)
            genome.set_trained_accuracy(accuracy)

            genomeFilename = os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.getGeneration(), genome.getUId()), GA_Constants.GENOME_FILENAME.format(self._gid, genome.getGeneration(), genome.getUId()))
            if not os.path.exists(genomeFilename):
                self.saveGenome(genomeFilename, genome)


    def _evaluateResult(self, jobDict):
        """
        Evaluate the result of the job.
        Overwrite the implementation in the JobAdministration class.
        @author: Markus Pfeil
        """
        assert type(jobDict) is dict

        if not self._checkAnn(jobDict['genomeUid'], jobDict['generation']):
            if jobDict['queue'] == 'clmedium':
                jobDict['queue'] = 'cllong'
                self._removeDirectory(jobDict['genomeUid'], jobDict['generation'])
                self._startJob(jobDict)
            elif jobDict['queue'] == 'cllong':
                jobDict['queue'] = 'clbigmem'
                self._removeDirectory(jobDict['genomeUid'], jobDict['generation'])
                self._startJob(jobDict)
            else:
                logging.info('***Could not train the neural network {} in generation {}***'.format(jobDict['genomeUid'], jobDict['generation']))
                assert False
        else:
            genomeFile = os.path.join(self._pathGA, jobDict['genomeFilename'])
            if os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                os.remove(genomeFile)
            jobDict["finished"] = True

        return True


    def generateBackup(self, generation=None, compression='bz2', compresslevel=9):
        """
        Create a backup in a tarfile of the whole data (generation is None) or for the given generation using the uids in the uidDic for the given generation.
        @author: Markus Pfeil
        """
        assert generation is None or type(generation) is int and 0 < generation
        assert generation is None or generation in self._genomesUidDic
        assert compression in ['bz2']
        assert compresslevel in range(1, 10)

        if generation is None:
             tarfilename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.PATTERN_TARFILE.format(self._gid, compression))
        else:
            tarfilename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.PATTERN_TARFILE_GENERATION.format(self._gid, generation, compression))

        if not os.path.exists(tarfilename):
            logging.debug('***Generate backup in file {}***'.format(tarfilename))
            tar = tarfile.open(tarfilename, 'w:{}'.format(compression), compresslevel=compresslevel)

            act_path = os.getcwd()
            os.chdir(self._pathGA)

            if generation is None:
                #Add all files and directories to the tarfile
                for f in os.listdir(self._pathGA):
                    try:
                        tar.add(f)
                    except tarfile.TarError:
                        logging.info('Can not add the {} file to archiv'.format(f))
            else:
                #Add only the files and directories for the given generation to the tarfile
                for uid in self._genomesUidDic[generation]:
                    path = GA_Constants.GENETIC_ALGORITHM_MODEL.format(generation, uid)
                    try:
                        tar.add(path)
                    except tarfile.TarError:
                        logging.info('Can not add the {} directory to archiv'.format(path))

                    #Add logs
                    log = os.path.join('Logs', GA_Constants.PATTERN_LOGFILE_GENETIC_ALGORITHM.format(self._gid))
                    try:
                        tar.add(log)
                    except tarfile.TarError:
                        logging.info('Can not add the log {} to archiv'.format(log))

                    logfile = os.path.join('Logs', 'LogfileTraining', GA_Constants.PATTERN_LOGFILE_TRANING_GENOME.format(self._gid, generation, uid))
                    try:
                        tar.add(logfile)
                    except tarfile.TarError:
                        logging.info('Can not add the logfile {} to archiv'.format(logfile))

                    joboutput = os.path.join('Logs', 'Joboutput', GA_Constants.PATTERN_JOBOUTPUT_TRANING_GENOME.format(self._gid, generation, uid))
                    try:
                        tar.add(joboutput)
                    except tarfile.TarError:
                        logging.info('Can not add the joboutput {} to archiv'.format(joboutput))

                    #Add not trained genomes files
                    genomeFiles = list(filter(lambda x: os.path.isfile(os.path.join(self._pathGA, x)) and x.startswith(GA_Constants.GENOME_FILENAME.format(self._gid, generation, 0).rsplit('_', maxsplit=3)[0]), os.listdir(self._pathGA)))
                    for f in genomeFiles:
                        try:
                            tar.add(f)
                        except tarfile.TarError:
                            logging.info('Can not add the genome file {} to archiv'.format(f))

            tar.close()
            os.chdir(act_path)


    def _generateGenomeFile(self, genome):
        """
        Save the ID and the generation of the genome in a file.
        @author: Markus Pfeil
        """
        annBestFilename = os.path.join(self._pathGA, ANN_Constants.ANN_GENETIC_ALGORITHM_BEST.format(self._gid))
        with open(annBestFilename, 'w') as f:
            f.write('ANN with the smallest loss error for genetic algorithm {:0>3d}\n'.format(self._gid))
            f.write('Generation: {:0>2d}\n'.format(genome.getGeneration()))
            f.write('ID: {:0>2d}\n'.format(genome.getUId()))
            f.write('Path: {}\n'.format(os.path.join(self._pathGA, GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.getGeneration(), genome.getUId()))))


    def readBestGenomeFile(self):
        """
        Read the generation, uid and path of the best network trained with the genetic algorithm.
        @author: Markus Pfeil
        """
        annBestFilename = os.path.join(self._pathGA, ANN_Constants.ANN_GENETIC_ALGORITHM_BEST.format(self._gid))
        generation = None
        uid = None
        path = None
        assert os.path.exists(annBestFilename) and os.path.isfile(annBestFilename)
        with open(annBestFilename, 'r') as f:
            for line in f.readlines():
                match_generation = re.search(r'Generation: (\d+)', line)
                match_id = re.search(r'ID: (\d+)', line)
                match_path = re.search(r'Path: (\S+)', line)
                if match_generation:
                    generation = int(match_generation.groups()[0])
                elif match_id:
                    uid = int(match_id.groups()[0])
                elif match_path:
                    path = match_path.groups()[0]

        assert not generation is None and not uid is None and not path is None
        return (generation, uid, path)

