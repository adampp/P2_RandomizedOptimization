import numpy as np
import matplotlib.pyplot as plt
import time


def plot_iter_curve(solver, problem, title, numTests, ylim=None, xlimBounds=None):
    
    fig, axes = plt.subplots(1, 1)
    axes = [axes]

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Fitness Value")
    
    startTime = time.time()
    scores = []
    randomStates = np.random.randint(0, 100, size=(numTests))
    for i in range(numTests):
        np.random.seed(randomStates[i])
        results = solver.solve(problem)
        scores.append(results[2])
        
    execTime = (time.time() - startTime) / numTests
        
    finalScores = []
    for i in range(numTests):
        finalScores.append(scores[i][-1])
        
    finalScores = np.array(finalScores)
    
        
    maxLen = 0
    for scoreArr in scores:
        if len(scoreArr) > maxLen:
            maxLen = len(scoreArr)
            
    scoreArr = np.zeros([numTests, maxLen])
    for i in range(len(scores)):
        if len(scores[i]) < maxLen:
            newArr = np.ones([maxLen]) * scores[i][-1]
            newArr[0:len(scores[i])] = scores[i]
            scores[i] = newArr
        
        scoreArr[i, :] = scores[i]
    
    if not problem.max:
        finalScores = finalScores * -1
        scoreArr = scoreArr * -1
    

    # Plot learning curve
    axes[0].grid()
    axes[0].plot(range(maxLen), scoreArr.mean(axis=0), '-', color="g",
                 label="Fitness Value")
    axes[0].fill_between(range(maxLen), scoreArr.mean(axis=0) - 1.00*scoreArr.std(axis=0),
                         scoreArr.mean(axis=0) + 1.00*scoreArr.std(axis=0), alpha=0.1,
                         color="g", label="68.2% Confidence Band")
                         
    axes[0].plot(range(maxLen), scoreArr.min(axis=0), '-', color="g", linewidth=0.6, 
                 label="Min/Max Fitness Value")
    axes[0].plot(range(maxLen), scoreArr.max(axis=0), '-', color="g", linewidth=0.6)
    axes[0].legend(loc="best")
    
    xlim = axes[0].get_xlim()
    if not xlimBounds is None:
        if xlim[0] > xlimBounds[0]:
            xlimBounds[0] = xlim[0]
        if xlim[1] < xlimBounds[1]:
            xlimBounds[1] = xlim[1]
    
    axes[0].set_xlim(xlimBounds)
    
    resultStr = f"minVal={np.min(finalScores)},\tmaxVal={np.max(finalScores)},\t\
meanVal={np.mean(finalScores)},\tmedianVal={np.median(finalScores)},\t\
68.2%LB={np.mean(finalScores)-1.00*np.std(finalScores)},\texecTime={execTime}"

    # fig.text(0.5, 0.5, 'Adam Pierce',
     # fontsize=62, color='gray',
     # ha='center', va='center', alpha=0.8)

    return plt, resultStr
