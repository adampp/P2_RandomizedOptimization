import mlrose_reborn as mlrose
import numpy as np
from sklearn import metrics

class RHCNN():
    plotTitle = "Randomized Hill Climb"
    fileName = "HillClimb"
    algorithm = 'random_hill_climb'
    gridSearch = {"learning_rate": [0.05, 0.1, 0.2, 0.4, 1.0], "restarts": [1, 2, 4, 10, 20, 40]}
    def __init__(self, maxIters = np.Inf, learnRate = 0.05, maxAttempts = 5, numRestarts = 0, randomState = None):
        self.maxIters = maxIters
        self.learnRate = learnRate
        self.maxAttempts = maxAttempts
        self.numRestarts = numRestarts
        self.randomState = randomState
        
    def createModel(self):
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                        algorithm = self.algorithm, max_iters = self.maxIters, \
                                        bias = True, is_classifier = True, learning_rate = self.learnRate, \
                                        early_stopping = False, clip_max = 1e10, max_attempts = self.maxAttempts, \
                                        restarts = self.numRestarts, curve = True, random_state = self.randomState)
    
    def solve(self, X, Y):
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                        algorithm = self.algorithm, max_iters = self.maxIters, \
                                        bias = True, is_classifier = True, learning_rate = self.learnRate, \
                                        early_stopping = False, clip_max = 1e10, max_attempts = self.maxAttempts, \
                                        restarts = self.numRestarts, curve = True, random_state = self.randomState)
        self.model.fit(X, Y)
        yPred = self.model.predict(X)
        score = metrics.accuracy_score(Y, yPred)
        return score
        
    def searchSolve(self, X, Y, updates, randomState = None):
        if "max_attempts" in updates:
            tempAttempts = updates["max_attempts"]
        else:
            tempAttempts = self.maxAttempts
            
        if "restarts" in updates:
            tempRestarts = updates["restarts"]
        else:
            tempRestarts = self.numRestarts
            
        if randomState is None:
            tempRandomState = self.randomState
        else:
            tempRandomState = randomState
            
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
                                        restarts = tempRestarts, curve = True, random_state = tempRandomState)
        self.model.fit(X, Y)
        yPred = self.model.predict(X)
        score = metrics.accuracy_score(Y, yPred)
        return score