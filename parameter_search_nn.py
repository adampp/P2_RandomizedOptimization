import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def parameter_search_nn(nn, X, Y, numTests):
    paramDict = nn.gridSearch
    
    fig, axes = plt.subplots(1, len(paramDict), figsize=(5.5*len(paramDict), 5))
    if len(paramDict) == 1:
        axes = [axes]
    axisIdx = 0
    title = f"Parameter Optimization for Neural Network" + "\n" + f"Solved with {nn.plotTitle} "
    fig.suptitle(title)
    
    setParams = {}
    resultStr = ""
    
    for key, value in paramDict.items():
        # axes[axisIdx].set_title(title)
        axes[axisIdx].set_xlabel(key)
        axes[axisIdx].set_ylabel("Fitness Value")
        
        scores = np.zeros((numTests, len(value)))
        randomStates = np.random.randint(0, 100, size=(numTests))
        paramIdx = 0
        for paramVal in value:
            subscores = np.array([])
            for i in range(numTests):
                tempParams = deepcopy(setParams)
                tempParams[key] = paramVal
                print(tempParams)
                results = nn.searchSolve(X, Y, tempParams, randomStates[i])
                subscores = np.block([subscores, results])
            scores[:, paramIdx] = subscores
            paramIdx += 1
        
        finalScores = scores.mean(axis=0) - 1.00*scores.std(axis=0)
        
        if type(value[0]) != int and type(value[0]) != float:
            xvalue = nn.scheduleString
        else:
            xvalue = value

        # Plot learning curve
        axes[axisIdx].grid()
        axes[axisIdx].plot(xvalue, scores.mean(axis=0), '-', color="g",
                     label=key)
        axes[axisIdx].fill_between(xvalue, scores.mean(axis=0) - 1.00*scores.std(axis=0),
                             scores.mean(axis=0) + 1.00*scores.std(axis=0), alpha=0.1,
                             color="g", label="68.2% Confidence Band")
        axes[axisIdx].legend(loc="best")
    
        resultStr = resultStr + "\t"+f"{key}: fitnessVal={np.max(finalScores)}, index={np.argmax(finalScores)}, param={value[np.argmax(finalScores)]}"
        setParams[key] = value[np.argmax(finalScores)]
        
        axisIdx += 1
        if axisIdx < len(paramDict):
            resultStr += "\n"

    # fig.text(0.5, 0.5, 'Adam Pierce',
     # fontsize=62, color='gray',
     # ha='center', va='center', alpha=0.8)

    return plt, resultStr
