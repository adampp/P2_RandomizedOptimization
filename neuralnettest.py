import mlrose_reborn as mlrose
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics

from parameter_search_nn import *
from plot_learning_curve import *
from plot_nn_learning_curve import *

from data_baby import *
from data_adult import *

from nn_hillclimb import *
from nn_simanneal import *
from nn_geneticalgo import *

randomState = 1
testSize = 0.15
kfolds = 7
maxIters = 3e5

dataSet = BabyData()
models = [0, 0, 0]

###########################
stepNum = 8
run = "learningcurve"
updateTxt = True
###########################

X, Y = dataSet.load()
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
    random_state = randomState)
    
print((float(np.sum(yTrain[:,0])) / float(len(yTrain))))

    
# models[0] = RHCNN(maxIters = maxIters, learnRate = 0.1, numRestarts = 0, randomState = 1)
# models[1] = SimAnnealNN(maxIters = maxIters, learnRate = 0.1, schedule = mlrose.GeomDecay(), randomState = 1)
# models[2] = GeneticAlgoNN( maxIters = maxIters, learnRate = 0.1, popSize = 200, mutationProb = 0.1, randomState = 1)

models[0] = RHCNN(maxIters = maxIters, learnRate = 1.0, numRestarts = 20, randomState = 1)
models[1] = SimAnnealNN(maxIters = maxIters, learnRate = 0.1, schedule = mlrose.ArithDecay(decay=0.0001, init_temp = 2.0), randomState = 1)
models[2] = GeneticAlgoNN( maxIters = maxIters, learnRate = 0.2, popSize = 200, mutationProb = 0.15, randomState = 1)

for i in [1]:
    nn = models[i]
    if run == "paramsearch":
        plt, resultStr = parameter_search_nn(nn, xTrain, yTrain, numTests=10)
        plt.savefig(f"plots/NN_{nn.fileName}_{stepNum}_ParamSearch.png", format='png')
        print(f"NN-ParamSearch-{stepNum}-{nn.fileName}:")
        print(resultStr)

        if updateTxt:
            with open(f"plots/NN_{nn.fileName}.txt", "a") as file:
                file.write("====================================================\n")
                file.write(f"ParamSearch-{stepNum}-{nn.fileName}:"+"\n")
                file.write(resultStr+"\n")
            
    elif run == "learningcurve":
        title = f"Learning Curve for {nn.plotTitle}"
        nn.createModel()
        plt, resultStr = plot_learning_curve(nn.model, title, xTrain, yTrain, cv = kfolds)
        plt.savefig(f"plots/NN_{nn.fileName}_{stepNum}_LearningCurve.png", format='png')
        print(f"NN-LearningCurve-{stepNum}-{nn.fileName}:")
        print(resultStr)

        if updateTxt:
            with open(f"plots/NN_{nn.fileName}.txt", "a") as file:
                file.write("====================================================\n")
                file.write(f"LearningCurve-{stepNum}-{nn.fileName}:"+"\n")
                file.write(resultStr+"\n")
        
        epochs = np.linspace(1e4, maxIters, 20)
        plot_nn_learning_curve(nn, title, xTrain, yTrain, cv=kfolds, testSize=testSize, ylim=None)
        plt.savefig(f"plots/NN_{nn.fileName}_{stepNum}_IterCurve.png",
            format='png')
