import mlrose_reborn as mlrose
import numpy as np


class FlipFlopProb():
    plotTitle = "FlipFlop"
    fileName = "FlipFlop"
    def __init__(self, length):
        
        self.length = length
        self.maxVal = 2
        self.fitness = mlrose.FlipFlop()
        self.initState = np.random.randint(self.maxVal, size=self.length)
        self.max = True
        self.problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = self.fitness, maximize = self.max, max_val = self.maxVal)
        
    def initGuess(self):
        pass
        # return np.random.randint(self.maxVal, size=self.length)
        return self.initState