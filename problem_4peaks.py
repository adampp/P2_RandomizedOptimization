import mlrose_reborn as mlrose
import numpy as np


class FourPeaksProb():
    plotTitle = "Four Peaks"
    fileName = "FourPeaks"
    def __init__(self, t_pct, length):
        self.maxVal = 2
        self.length = length
        self.initState = np.random.randint(self.maxVal, size=self.length)
        self.fitness = mlrose.FourPeaks(t_pct = t_pct)
        self.max = True
        self.problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = self.fitness, maximize = self.max, max_val = self.maxVal)
        
    def initGuess(self):
        pass
        # return np.random.randint(self.maxVal, size=self.length)
        return self.initState