import numpy as np
from knnModelBased import *
from sklearn import metrics

def main():
    #read data
    data = np.genfromtxt('../Data/iris.data',  delimiter=",", skip_header=False)

    X = data[:,0:4]
    y = data[:,4]

    #shuffle
    p = np.random.permutation(len(X))
    X = X[p]
    y = y[p]

    classes, y = np.unique(y, return_inverse=True)

    #train with all data
    representatives = train(X,y,0)
    predicted_labels = classifyAll(X,representatives)
    print("5 fold cross validation avg accuracy: {}".format(kfoldCrossValidation(X,y, 5)))

#     graph number of representatives for different values of erd

if __name__ == "__main__":
    main()