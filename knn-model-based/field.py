import numpy as np
from knnModelBased import *

def main():
    #read data
    data = np.loadtxt('../Data/fieldgoal.dat', usecols = (0,1))

    data = np.hsplit(data,2)
    X = data[0]
    y = data[1]


    #train with all data
    representatives = train(X,y)
    print(representatives)

if __name__ == "__main__":
    main()