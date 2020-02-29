import mlrose_reborn as mlrose
import numpy as np

class HillClimbSolver():
    plotTitle = "Random Hill Climb"
    fileName = "HillClimb"
    gridSearch = {"restarts": [0, 2, 4, 10, 20, 50], "max_attempts": [5, 10, 20, 50]}
    def __init__(self, maxIters = np.Inf, maxAttempts = 1, numRestarts = 0, randomState = None):
        self.maxIters = maxIters
        self.maxAttempts = maxAttempts
        self.numRestarts = numRestarts
        self.randomState = randomState
    
    def solve(self, aProb):
        self.results = mlrose.random_hill_climb(aProb.problem, max_iters = self.maxIters,
            restarts = self.numRestarts, max_attempts = self.maxAttempts, init_state = aProb.initGuess(), random_state = self.randomState,
            curve = True)
        return self.results
        
    def searchSolve(self, aProb, updates):
        if "max_attempts" in updates:
            tempAttempts = updates["max_attempts"]
        else:
            tempAttempts = self.maxAttempts
            
        if "restarts" in updates:
            tempRestarts = updates["restarts"]
        else:
            tempRestarts = self.numRestarts
            
        self.results = mlrose.random_hill_climb(aProb.problem, max_iters = self.maxIters,
            restarts = tempRestarts, init_state = aProb.initGuess(), random_state = self.randomState,
            max_attempts = tempAttempts, curve = True)
        return self.results
        