from scipy.spatial.distance import pdist, squareform
import numpy as np

def getDistanceMatrix(x):
    return squareform(pdist(x, 'euclidean'))

def getUngrouped(s):
    ind =  [ k for k in range(len(s)) if s[k] == 0 ]
    return ind

def train(X,y):
    m = X.shape[0]
    states = np.zeros(m)
    distMatrix = getDistanceMatrix(X)
    ungrouped = getUngrouped(states)


    representatives = list()
    #while there are examples to group
    while(len(ungrouped) > 0):
        maxNeighbourhood = list()
        tupleMaxNeighbourhood = None
        for i in ungrouped:
            #get distance from i to all other tuples
            distances = distMatrix[i]
            #sort distances
            sorted_distances = [d for d in sorted(enumerate(distances), key=lambda x:x[1])]
            #filter only those which has not been yet grouped
            sorted_distances = [d for d in sorted_distances if states[d[0]] == 0]
            #compute neigbourhood
            q = 0
            neighbourhood = list()
            while (q < len(sorted_distances)and y[sorted_distances[q][0]] == y[i]):
                neighbourhood.append(sorted_distances[q][0])
                q+=1
            if (len(neighbourhood) > len(maxNeighbourhood)):
                maxNeighbourhood = neighbourhood
                tupleMaxNeighbourhood = i
        representatives.append((tupleMaxNeighbourhood, maxNeighbourhood, y[tupleMaxNeighbourhood], distMatrix[tupleMaxNeighbourhood,maxNeighbourhood[-1]]))
        for i in maxNeighbourhood:
            states[i] = 1
        ungrouped = getUngrouped(states)
        print(len(representatives), len(representatives[len(representatives)-1][1]))
    return representatives

