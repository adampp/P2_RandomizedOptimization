import mlrose_reborn as mlrose
import numpy as np

class SimulatedAnnealingSolver():
    plotTitle = "Simulated Annealing"
    fileName = "SimAnneal"
    # gridSearch = {"schedule": [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()] , "max_attempts": [1, 5, 10]}
    # scheduleString = ["GeometricDecay", "ArithmeticDecay", "ExponentialDecay"]
    # gridSearch = {"schedule": [mlrose.GeomDecay(decay=0.999), mlrose.GeomDecay(decay=0.99), mlrose.GeomDecay(decay=0.96), mlrose.GeomDecay(decay=0.90)]}
    # scheduleString = ["decay 0.999", "decay 0.99", "decay 0.96", "decay 0.90"]
    # gridSearch = {"schedule": [mlrose.ArithDecay(init_temp=0.1, decay=0.96), mlrose.ArithDecay(init_temp=0.25, decay=0.96), mlrose.ArithDecay(init_temp=0.5, decay=0.96), mlrose.ArithDecay(init_temp=1.0, decay=0.96)]}
    # scheduleString = ["init 0.1", "init 0.25", "init 0.5", "init 1.0"]
    gridSearch = {"schedule": [mlrose.ArithDecay(init_temp=0.1, decay=0.99), mlrose.ArithDecay(init_temp=0.25, decay=0.99), mlrose.ArithDecay(init_temp=0.5, decay=0.99), mlrose.ArithDecay(init_temp=1.0, decay=0.99)]}
    scheduleString = ["init 0.1", "init 0.25", "init 0.5", "init 1.0"]
    def __init__(self, schedule = mlrose.GeomDecay(), maxIters = np.Inf, maxAttempts = 1,
        randomState = None):
        self.schedule = schedule
        self.maxIters = maxIters
        self.maxAttempts = maxAttempts
        self.randomState = randomState
    
    def solve(self, aProb):
        self.results = mlrose.simulated_annealing(aProb.problem, schedule = self.schedule, max_attempts = self.maxAttempts,
            max_iters = self.maxIters, init_state = aProb.initGuess(), random_state = self.randomState, curve = True)
        return self.results
        
    def searchSolve(self, aProb, updates):
        if "max_attempts" in updates:
            tempAttempts = updates["max_attempts"]
        else:
            tempAttempts = self.maxAttempts
            
        if "schedule" in updates:
            tempSchedule = updates["schedule"]
        else:
            tempSchedule = self.schedule
            
        self.results = mlrose.simulated_annealing(aProb.problem, schedule = tempSchedule, max_attempts = tempAttempts,
            max_iters = self.maxIters, init_state = aProb.initGuess(), random_state = self.randomState, curve = True)
        return self.results