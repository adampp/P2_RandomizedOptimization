import mlrose_reborn as mlrose
import numpy as np


class KnapsackProb():
    plotTitle = "Knapsack"
    fileName = "Knapsack"
    def __init__(self):
        self.weights = [10, 5, 2, 8, 15]
        self.values = [1, 2, 3, 4, 5]
        self.maxWeightPct = 0.5
        self.initState = np.array([0, 0, 0, 0, 0])
        
        self.length = len(self.weights)
        self.maxVal = 12
        self.fitness = mlrose.Knapsack(self.weights, self.values, self.maxWeightPct)
        self.max = True
        self.problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = self.fitness, maximize = self.max, max_val = self.maxVal)
        
    def initGuess(self):
        pass
        # return np.random.randint(self.maxVal, size=self.length)
        return self.initState