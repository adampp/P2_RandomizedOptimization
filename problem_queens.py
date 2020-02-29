import mlrose_reborn as mlrose
import numpy as np


class QueensProb():
    plotTitle = "Queens"
    fileName = "Queens"
    def __init__(self, length):
        
        self.length = length
        self.maxVal = length - 1
        self.fitness = mlrose.Queens()
        self.initState = np.random.randint(self.maxVal, size=self.length)
        self.max = False
        self.problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = self.fitness, maximize = self.max, max_val = self.maxVal)
        
    def initGuess(self):
        pass
        # return np.random.randint(self.maxVal, size=self.length)
        return self.initState