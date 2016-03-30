import numpy as np
from knnModelBased import *

def main():
    #read data
    data = np.genfromtxt('../Data/iris.data',  delimiter=",", skip_header=False)


    X = data[:,0:4]
    y = data[:,4]


    #train with all data
    representatives = train(X,y)

    print(representatives)
if __name__ == "__main__":
    main()