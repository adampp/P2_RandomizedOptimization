import mlrose_reborn as mlrose
import numpy as np

class MimicSolver():
    plotTitle = "MIMIC"
    fileName = "MIMIC"
    gridSearch = {"pop_size": [20, 50, 100, 200, 400], "keep_pct": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "max_attempts": [1, 5, 10], }
    def __init__(self, popSize = 200, keepProb = 0.2, maxIters = np.Inf, maxAttempts = 1,
        randomState = None):
        self.popSize = popSize
        self.keepProb = keepProb
        self.maxIters = maxIters
        self.maxAttempts = maxAttempts
        self.randomState = randomState
    
    def solve(self, aProb):
        self.results = mlrose.mimic(aProb.problem, pop_size = self.popSize, keep_pct = self.keepProb,
            max_attempts = self.maxAttempts, max_iters = self.maxIters, random_state = self.randomState,
            curve = True)
        return self.results
        
    def searchSolve(self, aProb, updates):
        if "max_attempts" in updates:
            tempAttempts = updates["max_attempts"]
        else:
            tempAttempts = self.maxAttempts
            
        if "pop_size" in updates:
            tempPopSize = updates["pop_size"]
        else:
            tempPopSize = self.popSize
            
        if "keep_pct" in updates:
            tempKeepPct = updates["keep_pct"]
        else:
            tempKeepPct = self.keepProb
            
        self.results = mlrose.mimic(aProb.problem, pop_size = tempPopSize, keep_pct = tempKeepPct,
            max_attempts = tempAttempts, max_iters = self.maxIters, random_state = self.randomState,
            curve = True)
        return self.results