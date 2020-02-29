import mlrose_reborn as mlrose
import numpy as np

class GeneticAlgoSolver():
    plotTitle = "Genetic Algorithm"
    fileName = "GeneticAlgo"
    gridSearch = {"pop_size": [20, 50, 100, 200, 400], "mutation_prob": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], "max_attempts": [1, 5, 10]}
    def __init__(self, popSize = 200, mutationProb = 0.1, maxIters = np.Inf, maxAttempts = 1,
        randomState = None):
        self.popSize = popSize
        self.mutationProb = mutationProb
        self.maxIters = maxIters
        self.maxAttempts = maxAttempts
        self.randomState = randomState
    
    def solve(self, aProb):
        self.results = mlrose.genetic_alg(aProb.problem, pop_size = self.popSize, mutation_prob = self.mutationProb,
            max_attempts = self.maxAttempts, max_iters = self.maxIters, random_state = self.randomState, curve = True)
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
            
        if "mutation_prob" in updates:
            tempMutProb = updates["mutation_prob"]
        else:
            tempMutProb = self.mutationProb
            
        
            
        self.results = mlrose.genetic_alg(aProb.problem, pop_size = tempPopSize, mutation_prob = tempMutProb,
            max_attempts = tempAttempts, max_iters = self.maxIters, random_state = self.randomState, curve = True)
        return self.results