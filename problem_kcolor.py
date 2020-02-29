import mlrose_reborn as mlrose
import numpy as np


class KColorsProb():
    plotTitle = "K Colors"
    fileName = "KColors"
    def __init__(self):
        # self.graph = [(2, 4), (4, 7), (3, 1), (5, 1), (2, 7), (6, 1), (0, 6), (1, 4), (7, 6)]
        # self.nodes = 8
        # self.maxVal = 2
        
        # self.graph = [(12, 13), (12, 14), (7, 11), (8, 4), (3, 5), (0, 10), (8, 12), (5, 4),
            # (1, 9), (0, 8), (6, 14), (7, 6), (6, 5), (13, 4), (2, 8), (1, 14), (9, 10),
            # (4, 11), (11, 6), (11, 1), (5, 11), (4, 7), (2, 13), (12, 10), (10, 5), (4, 10),
            # (11, 7), (13, 12), (9, 11), (11, 0), (0, 4), (8, 6), (12, 8), (5, 10), (8, 2),
            # (0, 1), (6, 11), (0, 3), (9, 0), (14, 10)]
        # self.nodes = 15
        # self.maxVal = 2
        
        self.graph = [(1, 2), (1, 9), (3, 6), (5, 7), (8, 2), (0, 7), (3, 7), (9, 7), (4, 3),
            (3, 8), (2, 6), (8, 7), (6, 3), (1, 7), (7, 2), (0, 9), (8, 3), (6, 5), (8, 6),
            (5, 1), (2, 4), (5, 3), (1, 4), (9, 5), (1, 3), (0, 1), (4, 6), (5, 6), (9, 6),
            (9, 2), (6, 0), (3, 0)]
        self.nodes = 10
        self.maxVal = 3
        
        self.edges = len(self.graph)
        self.length = self.edges
        self.initState = np.random.randint(self.maxVal, size=self.length)
        self.fitness = mlrose.MaxKColor(edges = self.graph)
        self.max = False
        self.problem = mlrose.DiscreteOpt(length = self.length, fitness_fn = self.fitness, maximize = self.max, max_val = self.maxVal)
        
    def initGuess(self):
        pass
        # return np.random.randint(self.maxVal, size=self.length)
        return self.initState