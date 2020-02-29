import mlrose_reborn as mlrose
import numpy as np
from sklearn import metrics

class GeneticAlgoNN():
    plotTitle = "Genetic Algorithm"
    fileName = "GeneticAlgo"
    algorithm = "genetic_alg"
    gridSearch = {"learning_rate": [0.05, 0.1, 0.2, 0.4, 1.0], "pop_size": [50, 100, 200, 400], "mutation_prob": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]}
    def __init__(self, maxIters = np.Inf, learnRate = 0.05, popSize = 200, mutationProb = 0.1, maxAttempts = 2,
        randomState = None):
        self.popSize = popSize
        self.mutationProb = mutationProb
        self.maxIters = maxIters
        self.maxAttempts = maxAttempts
        self.learnRate = learnRate
        self.randomState = randomState
        
    def createModel(self):
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                algorithm = self.algorithm, max_iters = self.maxIters, \
                                bias = True, is_classifier = True, learning_rate = self.learnRate, \
                                early_stopping = False, clip_max = 1e10, max_attempts = self.maxAttempts, \
                                pop_size = self.popSize, mutation_prob = self.mutationProb, \
                                curve = True, random_state = self.randomState)
    
    def solve(self, X, Y):
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                algorithm = self.algorithm, max_iters = self.maxIters, \
                                bias = True, is_classifier = True, learning_rate = self.learnRate, \
                                early_stopping = False, clip_max = 1e10, max_attempts = self.maxAttempts, \
                                pop_size = self.popSize, mutation_prob = self.mutationProb, \
                                curve = True, random_state = self.randomState)
        self.model.fit(X, Y)
        yPred = self.model.predict(X)
        score = metrics.accuracy_score(Y, yPred)
        return score
        
    def searchSolve(self, X, Y, updates, randomState = None):
        if "max_attempts" in updates:
            tempAttempts = updates["max_attempts"]
        else:
            tempAttempts = self.maxAttempts
            
        if "pop_size" in updates:
            tempPopSize = updates["pop_size"]
        else:
            tempPopSize = self.popSize
            
        if "mutation_prob" in updates:
            tempMutProb = updates["mutation_prob"]
        else:
            tempMutProb = self.mutationProb
            
        if "learning_rate" in updates:
            tempLearnRate = updates["learning_rate"]
        else:
            tempLearnRate = self.learnRate
            
        if "max_iters" in updates:
            tempMaxIters = updates["max_iters"]
        else:
            tempMaxIters = self.maxIters
            
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                algorithm = self.algorithm, max_iters = tempMaxIters, \
                                bias = True, is_classifier = True, learning_rate = tempLearnRate, \
                                early_stopping = True, clip_max = 1e10, max_attempts = tempAttempts, \
                                pop_size = tempPopSize, mutation_prob = tempMutProb, \
                                curve = True, random_state = randomState)
        self.model.fit(X, Y)
        yPred = self.model.predict(X)
        score = metrics.accuracy_score(Y, yPred)
        return score