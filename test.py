import mlrose_reborn as mlrose
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_iter_curve import *
from parameter_search import *

from problem_4peaks import *
from problem_kcolor import *
from problem_knapsack import *
from problem_flipflop import *
from problem_queens import *

from solver_hillclimb import *
from solver_simanneal import *
from solver_geneticalgo import *
from solver_mimic import *

probs = [0, 0, 0]
solvers = [0, 0, 0, 0]
tunedsolvers = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
np.random.seed(1)

###################################
maxAttempts = 10
maxIters = 1e5
stepNum = 6
probIdx = 0
useTuned = True
run = 'itercurve'
updateTxt = True
###################################

probs[0] = FourPeaksProb(0.1, 30)
probs[1] = KColorsProb()
probs[2] = QueensProb(8)

solvers[0] = GeneticAlgoSolver(popSize = 200, mutationProb = 0.1, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
solvers[1] = HillClimbSolver(numRestarts = 5, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
solvers[2] = MimicSolver(popSize = 200, keepProb = 0.2, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
solvers[3] = SimulatedAnnealingSolver(schedule = mlrose.GeomDecay(), maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)

tunedsolvers[0][0] = GeneticAlgoSolver(popSize = 400, mutationProb = 0.2, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
tunedsolvers[0][1] = GeneticAlgoSolver(popSize = 400, mutationProb = 0.25, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
tunedsolvers[0][2] = GeneticAlgoSolver(popSize = 400, mutationProb = 0.1, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)

tunedsolvers[1][0] = HillClimbSolver(numRestarts = 50, maxIters = maxIters, maxAttempts = 50, randomState = None)
tunedsolvers[1][1] = HillClimbSolver(numRestarts = 50, maxIters = maxIters, maxAttempts = 50, randomState = None)
tunedsolvers[1][2] = HillClimbSolver(numRestarts = 50, maxIters = maxIters, maxAttempts = 50, randomState = None)

tunedsolvers[2][0] = MimicSolver(popSize = 400, keepProb = 0.2, maxIters = maxIters, maxAttempts = 5, randomState = None)
tunedsolvers[2][1] = MimicSolver(popSize = 400, keepProb = 0.1, maxIters = maxIters, maxAttempts = 5, randomState = None)
tunedsolvers[2][2] = MimicSolver(popSize = 400, keepProb = 0.15, maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)

tunedsolvers[3][0] = SimulatedAnnealingSolver(schedule = mlrose.GeomDecay(decay=0.96, init_temp=0.25), maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
tunedsolvers[3][1] = SimulatedAnnealingSolver(schedule = mlrose.GeomDecay(decay=0.99, init_temp=1.0), maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)
tunedsolvers[3][2] = SimulatedAnnealingSolver(schedule = mlrose.GeomDecay(decay=0.96, init_temp=0.25), maxIters = maxIters, maxAttempts = maxAttempts, randomState = None)

for i in [2]:
    for j in [1]:
        aProb = probs[i]
        if useTuned:
            aSolver = tunedsolvers[j][i]
        else:
            aSolver = solvers[j]
            
        if run == 'paramsearch':
            plt, resultStr = parameter_search(aSolver, aProb, numTests=10)
            plt.savefig(f"plots/{aProb.fileName}_{aSolver.fileName}_{stepNum}_ParamSearch.png", format='png')
            print(f"ParamSearch-{stepNum}-{aProb.fileName}-{aSolver.fileName}:")
            print(resultStr)

            if updateTxt:
                with open(f"plots/{aProb.fileName}_{aSolver.fileName}.txt", "a") as file:
                    file.write("====================================================\n")
                    file.write(f"_ParamSearch-{stepNum}-{aProb.fileName}-{aSolver.fileName}:"+"\n")
                    file.write(resultStr+"\n")
        
        elif run == 'itercurve':
            title = f"{aSolver.plotTitle} on the {aProb.plotTitle} Problem"
            plt, resultStr = plot_iter_curve(aSolver, aProb, title, numTests=10, ylim=None, xlimBounds=[0, 1500])
            plt.savefig(f"plots/{aProb.fileName}_{aSolver.fileName}_{stepNum}_IterCurve.png", format='png')
            print(f"IterCurve-{stepNum}-{aProb.fileName}-{aSolver.fileName}:")
            print(resultStr)

            if updateTxt:
                with open(f"plots/{aProb.fileName}_{aSolver.fileName}.txt", "a") as file:
                    file.write("====================================================\n")
                    file.write(f"IterCurve-{stepNum}-{aProb.fileName}-{aSolver.fileName}:"+"\n")
                    file.write(resultStr+"\n")
        
        else:
            print('ERROR: Unrecognized test to run')
                
# solver first?
# for aProb in probs:
    # for aSolver in [solvers[0]]:
        # title = f"{aSolver.plotTitle} on the {aProb.plotTitle} Problem"
        # plt, resultStr = plot_iter_curve(aSolver, aProb, title, numTests=10, ylim=None, xlimBounds=[0, 1500])
        # plt.savefig(f"plots/{aSolver.fileName}_{aProb.fileName}_IterCurve.png", format='png')
        # print(f"IterCurve-{stepNum}-{aSolver.fileName}-{aProb.fileName}:")
        # print(resultStr)

        # if updateTxt:
            # with open(f"plots/{aSolver.fileName}_{aProb.fileName}.txt", "a") as file:
                # file.write("====================================================\n")
                # file.write(f"IterCurve-{stepNum}-{aSolver.fileName}-{aProb.fileName}:"+"\n")
                # file.write(resultStr+"\n")