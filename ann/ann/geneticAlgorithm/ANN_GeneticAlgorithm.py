#!/usr/bin/env python
# -*- coding: utf8 -*

import os
import shutil
import re
import logging
import bz2
import tarfile
import pickle

import neshCluster.constants as NeshCluster_Constants
from neshCluster.JobAdministration import JobAdministration
import metos3dutil.metos3d.constants as Metos3d_Constants
import ann.network.constants as ANN_Constants
import ann.geneticAlgorithm.constants as GA_Constants
import ann.geneticAlgorithm.geneticAlgorithm as Evolver
import ann.geneticAlgorithm.geneticAlgorithmRechenberg as EvolverRechenberg


class ANN_GeneticAlgorithm(JobAdministration):
    """
    Class for the organisation of the ANN training using a genetic algorithm.
    @author: Markus Pfeil
    """
    
    def __init__(self, gid=None, algorithm='GeneticAlgorithm', generations=10, populationSize=30, metos3dModel='N'):
        """
        Initialisation of the genetic algorithm.
        @author: Markus Pfeil
        """
        assert gid is None or type(gid) is int and 0 <= gid
        assert algorithm in GA_Constants.ALGORITHMS
        assert type(generations) is int and 0 < generations
        assert type(populationSize) is int and 0 < populationSize
        assert metos3dModel in Metos3d_Constants.METOS3D_MODELS

        JobAdministration.__init__(self)

        if gid is None:
            self._generateGid()
        else:
            path = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(gid))
            assert os.path.exists(path) and os.path.isdir(path)
            self._gid = gid

        self._algorithm = algorithm
        self._generations = generations
        self._populationSize = populationSize
        self._metos3dModel = metos3dModel


    def _generateGid(self):
        """
        Generate a unique gid and create directory.
        @author: Markus Pfeil
        """
        self._gid = 0
        path = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))

        while(os.path.exists(path) and os.path.isdir(path)):
            self._gid = self._gid + 1
            path = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))
        os.makedirs(path, exist_ok=False)


    def run(self, config={}, gaParameter={}):
        """
        Run the training of an ANN using the genetic algorithm.
        @author: Markus Pfeil
        """
        assert type(config) is dict 
        assert type(gaParameter) is dict
        
        logging.info('***Run the genetic algorithm with gid {:d} for {:d} generations with a population size of {:d}***'.format(self._gid, self._generations, self._populationSize))
        
        self._initEvolver(gaParameter=gaParameter)
        
        self._init()
        self._evolve(config=config)


    def _initEvolver(self, allPossibleGenes=GA_Constants.all_possible_genes, gaParameter={}):
        """
        Initialisation of the evolver.
        @author: Markus Pfeil
        """
        assert type(allPossibleGenes) is dict
        assert type(gaParameter) is dict

        if self._algorithm == 'GeneticAlgorithm':
            if len(gaParameter) == 3:
                assert 'retain' in gaParameter and type(gaParameter['retain']) is float and 0.0 < gaParameter['retain'] and gaParameter['retain'] < 1.0
                assert 'random_select' in gaParameter and type(gaParameter['random_select']) is float and 0.0 < gaParameter['random_select'] and gaParameter['random_select'] < 1.0
                assert 'mutate_chance' in gaParameter and type(gaParameter['mutate_chance']) is float and 0.0 < gaParameter['mutate_chance'] and gaParameter['mutate_chance'] < 1.0

                evolverParameter = gaParameter
            else:
                evolverParameter = GA_Constants.GA_Config_GeneticAlgorithm_Default

            logging.info('***Use probabilities for genetic algorithm: retain {:f} random_select {:f} and mutate_change {:f}***'.format(evolverParameter['retain'], evolverParameter['random_select'], evolverParameter['mutate_chance']))
            
            self._retain_length = int(self._populationSize * evolverParameter['retain'])
            self._evolver = Evolver.GeneticAlgorithm(allPossibleGenes, populationSize = self._populationSize, retain = evolverParameter['retain'], random_select = evolverParameter['random_select'], mutate_chance = evolverParameter['mutate_chance'])
        elif self._algorithm == 'Rechenberg':
            if len(gaParameter) == 2:
                assert 'alpha' in gaParameter and type(modelparameter['alpha']) is float and 0.0 < modelparameter['alpha']
                assert 'offspring' in gaParameter and type(modelparameter['offspring']) is int and 0 < modelparameter['offspring']

                evolverParameter = gaParameter
            else:
                evolverParameter = GA_Constants.GA_Config_Rechenberg_Default

            logging.info('***Used parameter for evolutionary strategy (Rechenberg): alpha {:f} and offspring {:d}***'.format(evolverParameter['alpha'], evolverParameter['offspring']))

            self._evolver = EvolverRechenberg.GeneticAlgorithm(all_possible_genes = allPossibleGenes, populationSize = self._populationSize, alpha = evolverParameter['alpha'], offspring = evolverParameter['offspring'])
        else:
            assert False, 'Not implemented genetic algorithm {:s}'.format(self._algorithm)


    def _init(self):
        """
        Initialisation of the training process using the genetic algorithm
        @author: Markus Pfeil
        """
        self._genomes = []
        self._uidDic = {}
        self._startGeneration = 1
        
        #Read the already trained genomes
        self._readTrainedGeneration()

        if self._startGeneration == 1 and len(self._genomes) == 0:
            self._genomes = self._evolver.create_population(self._populationSize)


    def _readTrainedGeneration(self):
        """
        Read the generations already trained.
        @author: Markus Pfeil
        """
        checkNextGeneration = True
        while checkNextGeneration and self._startGeneration < self._generations + 1:
            (checkNextGeneration, genomes) = self._readGenomesGeneration()
            if self._algorithm == 'GeneticAlgorithm':
                if self._startGeneration == 1:
                    self._genomes = genomes
                else:
                    self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)[:self._retain_length] + genomes
            elif self._algorithm == 'Rechenberg':
                if self._startGeneration == 1:
                    self._genomes = genomes
                elif self._startGeneration == 2:
                    self._genomes = sorted(self._genomes, key=lambda x: x.accuracy) + genomes
                else:
                    self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)[:self._populationSize] + genomes
            else:
                assert False

            if checkNextGeneration:
                self._updateUidList()
                self._startGeneration += 1


    def _readGenomesGeneration(self):
        """
        Read the genomes for one generation.
        Check, if all genomes for this generation are already trained.
        @author: Markus Pfeil
        """
        logging.debug('***Check generation {:d} for gid {:d} and population size {:d}***'.format(self._startGeneration, self._gid, self._populationSize))

        checkGeneration = True
        genomes = []

        #Read the genomes in the directories for the given generation
        genomesDirectories = list(filter(lambda x: os.path.isdir(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), x)) and x.startswith(GA_Constants.GENETIC_ALGORITHM_MODEL.format(self._startGeneration, 0).rsplit('_', maxsplit=1)[0]), os.listdir(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid)))))

        if len(genomesDirectories) == 0:
            checkGeneration = False
        elif self._algorithm == 'Rechenberg' and len(genomesDirectories) != self._populationSize:
            checkGeneration = False

        for dic in genomesDirectories:
            genomeUid = int(dic.rsplit('_', maxsplit=1)[1])

            if self._checkAnn(genomeUid, self._startGeneration):
                #Training of the genomes is already finished
                genomeFile = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(self._startGeneration, genomeUid), GA_Constants.GENOME_FILENAME.format(self._gid, self._startGeneration, genomeUid))
                if os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                    with open(genomeFile, 'rb') as f:
                        genome = pickle.load(f)
                        accuracy = self._readGenomeAccuracy(genome)
                        genome.set_trained_accuracy(accuracy)
                        increaseId = not self._containUid(genomeUid)
                        self._evolver.add_genome(genome, increaseId=increaseId)
                        genomes.append(genome)
            else:
                #Genome has to be trained so read the original genome file
                checkGeneration = False
                genomeFile = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENOME_FILENAME.format(self._gid, self._startGeneration, genomeUid))
                if os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                    with open(genomeFile, 'rb') as f:
                        genome = pickle.load(f)
                        increaseId = not self._containUid(genomeUid)
                        self._evolver.add_genome(genome, increaseId=increaseId)
                        genomes.append(genome)

                if os.path.exists(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), dic)) and os.path.isdir(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), dic)):
                    shutil.rmtree(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), dic))

        #Read the genomes without directory (so far, the training of the neural network using this genome has not been started)
        genomesFiles = list(filter(lambda x: os.path.isfile(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), x)) and x.startswith(GA_Constants.GENOME_FILENAME.format(self._gid, self._startGeneration, 0).rsplit('_', maxsplit=2)[0]), os.listdir(os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid)))))

        for genomeFile in genomesFiles:
            matchUid = re.search(GA_Constants.RE_PATTERN_GENOME_FILENAME, genomeFile)
            if matchUid:
                genomeUid = int(matchUid.groups()[2])
            else:
                assert False

            genomeDir = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(self._startGeneration, genomeUid))
            genomeFile = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENOME_FILENAME.format(self._gid, self._startGeneration, genomeUid))
            if not (os.path.exists(genomeDir) and os.path.isdir(genomeDir)) and os.path.exists(genomeFile) and os.path.isfile(genomeFile):
                checkGeneration = False
                with open(genomeFile, 'rb') as f:
                    genome = pickle.load(f)
                    increaseId = not self._containUid(genomeUid)
                    self._evolver.add_genome(genome, increaseId=increaseId)
                    genomes.append(genome)

        return (checkGeneration, genomes)


    def _evolve(self, config={}):
        """
        Evolve the popultion over the given number of generations.
        @author: Markus Pfeil
        """
        assert type(config) is dict 
        
        for generation in range(self._startGeneration, self._generations + 1):
            logging.info('***Now in generation {:d} of {:d}***'.format(generation, self._generations))

            self._updateUidList()
            logging.info('***List of uids in this generation: {}***'.format(self._uidDic[generation]))
            
            # Train and get accuracy for networks/genomes.
            self._trainGenomes(config=config)

            # Get the average accuracy for this generation and print the genomes.
            logging.info('***Generation average: {:.5e}***'.format(self._getAverageAccuracy()))
            self._printGenomes()

            # Evolve, except on the last iteration.
            if i != generations - 1:
                # Evolve!
                logging.info('***Evolve genomes for generation {:d}***'.format(generation + 1))
                self._genomes = evolver.evolve(self._genomes)

            #Create data backup for the generation
            logging.info('***Generate backup of the generation {:d}***'.format(generation))
            self.generateBackup(generation=generation)

        # Sort our final population according to performance.
        self._genomes = sorted(self._genomes, key=lambda x: x.accuracy)

        # Print out the networks/genomes.
        logging.info('***Sorted list of networks/genomes***')
        self._printGenomes()

        #Save file with the id and generation of the best ANN
        self._generateGenomeFile(self._genomes[0])

        #Create data backup for the whole calculation
        self.generateBackup()


    def _updateUidList(self):
        """
        Update the list of uids using the list of genomes.
        @author: Markus Pfeil
        """
        for genome in self._genomes:
            generation = genome.getGeneration()
            uid = genome.getUId()

            #Test, if the uid is already present in the uidDic
            uidKey = []
            for key in self._uidDic:
                if uid in self._uidDic[key]:
                    uidKey.append(key)

            if not generation in uidKey:
                try:
                    if not generation in self._uidDic:
                        self._uidDic[generation] = [uid]
                    else:
                        self._uidDic[generation].append(uid)
                        self._uidDic[generation] = sorted(self._uidDic[generation])
                except KeyError:
                    self._uidDic[generation] = [uid]


    def _containUid(self, uid):
        """
        Check, if the dictionary uidDic contains the given uid as value
        @author: Markus Pfeil
        """
        assert type(uid) is int and 0 <= uid

        checkUid = False
        for key in self._uidDic:
            if uid in self._uidDic:
                checkUid = True

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
                annPath = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))
                genomeFilename = GA_Constants.GENOME_FILENAME.format(self._gid, genome.getGeneration(), genome.getUId())
                if not (os.path.exists(os.path.join(annPath, genomeFilename)) and os.path.isfile(os.path.join(annPath, genomeFilename))):
                    self.saveGenome(os.path.join(annPath, genomeFilename), genome)

                #Set queue for the training
                if genome.getMaxepoches() < 4000 or "trainingSize" in config and config["trainingSize"] <= 500:
                    queue='clmedium'
                else:
                    queue='cllong'

                # Create directory for the joboutput
                pathLogs = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), 'Logs', 'Joboutput')
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
                jobDict['jobFilename'] = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.PATTERN_JOBFILE_TRANING_GENOME.format(self._gid, genome.getGeneration(), genome.getUId()))
                jobDict['jobname'] = 'GA_{}_{}_TrainingGenome'.format(self._gid, genome.getUId())
                jobDict['joboutput'] = os.path.join(pathLogs, GA_Constants.PATTERN_JOBOUTPUT_TRANING_GENOME.format(self._gid, genome.getGeneration(), genome.getUId()))
                jobDict['programm'] = os.path.join(NeshCluster_Constants.PYTHON_PATH, 'ann', 'ann', 'geneticAlgorithm', programm)
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

            genomeFilename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.getGeneration(), genome.getUId()), GA_Constants.GENOME_FILENAME.format(self._gid, genome.getGeneration(), genome.getUId()))
            if not os.path.exists(genomeFilename):
                self.saveGenome(genomeFilename, genome)


    def _evaluateResult(self, jobDict):
        """
        Evaluate the result of the job.
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
            genomeFile = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), jobDict['genomeFilename'])
            os.remove(genomeFile)
            jobDict["finished"] = True

        return True


    def _checkAnn(self, genomeUid, generation):
        """
        Check, if the directory includes the three files for the ann and the file for the gnome.
        @author: Markus Pfeil
        """
        assert type(genomeUid) is int and 0 <= genomeUid
        assert type(generation) is int and 0 < generation
        
        logging.info('***Check ANN (gid: {:d}, generation: {:d}, genomeUid: {:d})***'.format(self._gid, generation, genomeUid))
        path = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(generation, genomeUid))
        check_path = os.path.exists(path) and os.path.isdir(path)
        check_architecture = os.path.exists(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(genomeUid))) and os.path.isfile(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_ARCHITECTURE.format(genomeUid)))
        check_training = os.path.exists(os.path.join(path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(genomeUid))) and os.path.isfile(os.path.join(path, ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(genomeUid)))
        check_weights = os.path.exists(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(genomeUid))) and os.path.isfile(os.path.join(path, ANN_Constants.ANN_FILENAME_SET_WEIGHTS.format(genomeUid)))

        check_genome = os.path.exists(os.path.join(path, GA_Constants.GENOME_FILENAME.format(self._gid, generation, genomeUid))) and os.path.isfile(os.path.join(path, GA_Constants.GENOME_FILENAME.format(self._gid, generation, genomeUid)))

        logging.debug('***Check ANN (gid: {:d}, generation: {:d}, genomeUid: {:d})***\nCheck path: {}\nCheck architecture: {}\nCheck training: {}\nCheck weights: {}\nCheck genome file: {}'.format(self._gid, generation, genomeUid, check_path, check_architecture, check_training, check_weights, check_genome))
        return check_path and check_architecture and check_training and check_weights and check_genome


    def _removeDirectory(self, genomeUid, generation):
        """
        Remove the directory of the ANN.
        @author: Markus Pfeil
        """
        assert type(genomeUid) is int and 0 <= genomeUid
        assert type(generation) is int and 0 < generation

        path = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(generation, genomeUid))
        logging.info('***Remove directory for the ANN in generation {:d} with id {:d}: {}'.format(generation, genomeUid, path))

        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)


    def saveGenome(self, genomeFilename, genome):
        """
        Save genome in a file
        @author: Markus Pfeil
        """
        assert not os.path.exists(genomeFilename)

        with open(genomeFilename, 'wb') as f:
            pickle.dump(genome, f, pickle.HIGHEST_PROTOCOL)


    def _readGenomeAccuracy(self, genome):
        """
        Read the loss of the training from the training monitor file.
       @author: Markus Pfeil
        """
        filename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid), GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.generation, genome.u_ID), ANN_Constants.ANN_FILENAME_TRAINING_MONITOR.format(genome.u_ID))
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


    def _printGenomes(self):
        """
        Print the list of genomes.
        """
        logging.info('-'*80)

        for genome in self._genomes:
            genome.print_genome()

        logging.info('-'*80)


    def generateBackup(self, generation=None, compression='bz2', compresslevel=9):
        """
        Create a backup in a tarfile of the whole data (generation is None) or for the given generation using the uids in the uidDic for the given generation.
        @author: Markus Pfeil
        """
        assert generation is None or type(generation) is int and 0 < generation
        assert generation is None or generation in self._uidDic
        assert compression in ['bz2']
        assert compresslevel in range(1, 10)

        if generation is None:
             tarfilename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.PATTERN_TARFILE.format(self._gid, compression))
        else:
            tarfilename = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.PATTERN_TARFILE_GENERATION.format(self._gid, generation, compression))

        if not os.path.exists(tarfilename):
            logging.debug('***Generate backup in file {}***'.format(tarfilename))
            tar = tarfile.open(tarfilename, 'w:{}'.format(compression), compresslevel=compresslevel)
            tarpath = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))

            act_path = os.getcwd()
            os.chdir(tarpath)

            if generation is None:
                #Add all files and directories to the tarfile
                for f in os.listdir(tarpath):
                    try:
                        tar.add(f)
                    except tarfile.TarError:
                        logging.info('Can not add the {} file to archiv'.format(f))
            else:
                #Add only the files and directories for the given generation to the tarfile
                for uid in self._uidDic[generation]:
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
                    genomeFiles = list(filter(lambda x: os.path.isfile(os.path.join(tarpath, x)) and x.startswith(GA_Constants.GENOME_FILENAME.format(self._gid, generation, 0).rsplit('_', maxsplit=3)[0]), os.listdir(tarpath)))
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
        annPath = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))
        annBestFilename = os.path.join(annPath, ANN_Constants.ANN_GENETIC_ALGORITHM_BEST.format(self._gid))
        with open(annBestFilename, 'w') as f:
            f.write('ANN with the smallest loss error for genetic algorithm {:0>3d}\n'.format(self._gid))
            f.write('Generation: {:0>2d}\n'.format(genome.generation))
            f.write('ID: {:0>2d}\n'.format(genome.getUId()))
            f.write('Path: {}\n'.format(os.path.join(annPath, GA_Constants.GENETIC_ALGORITHM_MODEL.format(genome.generation, genome.getUId()))))


    def readBestGenomeFile(self):
        """
        Read the generation, uid and path of the best network trained with the genetic algorithm.
        @author: Markus Pfeil
        """
        annPath = os.path.join(ANN_Constants.PATH, GA_Constants.GENETIC_ALGORITHM_DIRECTORY, GA_Constants.GENETIC_ALGORITHM.format(self._gid))
        annBestFilename = os.path.join(annPath, ANN_Constants.ANN_GENETIC_ALGORITHM_BEST.format(self._gid))
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

