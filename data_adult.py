from math import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import random

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            this_xs = this_xs.reindex(np.random.permutation(this_xs.index))

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = pd.concat(xs)
    ys = pd.Series(data=np.concatenate(ys),name='target')

    return xs,ys

class AdultData:
    def __init__(self):
        self.filepath = "data/adult.data"
        self.delimiter = ', '
        self.outputMap = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}
        self.labels=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
        self.numerics=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        self.plotTitle = "Adult Dataset"
        self.plotFileName = "Adult"
    
    def loadBalanced(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        # idxs = random.sample_without_replacement(n_population=len(data), n_samples=ceil(len(data)*0.2), random_state = 1)
        # data = data.iloc[idxs, :]
        # X = data.iloc[:,0:-1]
        
        X, Y = balanced_subsample(data.iloc[:,0:-1], data.iloc[:,-1], 0.2)
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(len(self.numerics)):
            X[self.numerics[i]] = numericScaler.fit_transform(pd.DataFrame(X[self.numerics[i]]))
        
        
        return X, Y
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        idxs = random.sample_without_replacement(n_population=len(data), n_samples=ceil(len(data)*0.15), random_state = 1)
        data = data.iloc[idxs, :]
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(len(self.numerics)):
            X[self.numerics[i]] = numericScaler.fit_transform(pd.DataFrame(X[self.numerics[i]]))
        
        data.iloc[:,0:-1] = X
        
        X = data.iloc[:,0:-1]
        Y = data.iloc[:,-1]
        
        
        return X, Y