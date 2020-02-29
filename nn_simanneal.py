import mlrose_reborn as mlrose
import numpy as np
from sklearn import metrics

class SimAnnealNN():
    plotTitle = "Simulated Annealing"
    fileName = "SimAnneal"
    algorithm = 'simulated_annealing'
    # gridSearch = {"learning_rate": [0.05, 0.1, 0.2, 0.4, 1.0], "schedule": [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]}
    # scheduleString = ["GeometricDecay", "ArithmeticDecay", "ExponentialDecay"]
    # gridSearch = {"schedule": [mlrose.ArithDecay(decay=0.00001), mlrose.ArithDecay(decay=0.0001), mlrose.ArithDecay(decay=0.001), mlrose.ArithDecay(decay=0.01), mlrose.ArithDecay(decay=0.1)]}
    # scheduleString = ["decay 0.00001", "decay 0.0001", "decay 0.001", "decay 0.01", "decay 0.1"]
    gridSearch = {"schedule": [mlrose.ArithDecay(init_temp=0.25, decay=0.0001), mlrose.ArithDecay(init_temp=0.5, decay=0.0001),
        mlrose.ArithDecay(init_temp=1.0, decay=0.0001), mlrose.ArithDecay(init_temp=2.0, decay=0.0001), mlrose.ArithDecay(init_temp=5.0, decay=0.0001)]}
    scheduleString = ["init 0.25", "init 0.5", "init 1.0", "init 2.0", "init 5.0"]
    def __init__(self, maxIters = np.Inf, learnRate = 0.05, schedule = mlrose.GeomDecay(), maxAttempts = 2, randomState = None):
        self.learnRate = learnRate
        self.schedule = schedule
        self.maxIters = maxIters
        self.maxAttempts = maxAttempts
        self.randomState = randomState
        
    def createModel(self):
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                algorithm = self.algorithm, max_iters = self.maxIters, \
                                bias = True, is_classifier = True, learning_rate = self.learnRate, \
                                early_stopping = False, clip_max = 1e10, max_attempts = self.maxAttempts, \
                                schedule = self.schedule, curve = True, random_state = self.randomState)
    
    def solve(self, X, Y):
        self.model = mlrose.NeuralNetwork(hidden_nodes = [5], activation = 'relu', \
                                algorithm = self.algorithm, max_iters = self.maxIters, \
                                bias = True, is_classifier = True, learning_rate = self.learnRate, \
                                early_stopping = False, clip_max = 1e10, max_attempts = self.maxAttempts, \
                                schedule = self.schedule, curve = True, random_state = self.randomState)
        self.model.fit(X, Y)
        yPred = self.model.predict(X)
        score = metrics.accuracy_score(Y, yPred)
        return score
        
    def searchSolve(self, X, Y, updates, randomState = None):
        if "max_attempts" in updates:
            tempAttempts = updates["max_attempts"]
        else:
            tempAttempts = self.maxAttempts
            
        if "schedule" in updates:
            tempSchedule = updates["schedule"]
        else:
            tempSchedule = self.schedule
            
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
                                restarts = 10, schedule = tempSchedule, curve = True, \
                                random_state = randomState)
        self.model.fit(X, Y)
        yPred = self.model.predict(X)
        score = metrics.accuracy_score(Y, yPred)
        return score