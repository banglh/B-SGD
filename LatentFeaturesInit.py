import numpy as np
import scipy.sparse as sparse

def initParamsRandom1(Nu, Ni, K, rmin, rmax):
    print("Starting random initialization ...")
    # Randomly initialize P and Q
    P = np.random.random((Nu, K))
    Q = np.random.random((K, Ni))
    
    # Find max element of P.Q without first column of P and first row of Q
    maxE1 = maxElement1(P, Q)
    
    # recalculate P, Q
    alpha = np.sqrt((rmax - 1) / maxE1)
    P = P * alpha
    Q = Q * alpha
    P[:, 0] = 1
    Q[0, :] = 1
    
    print("Finished Random initialization!")
    return P, Q

def initParamsBaseline(Rtrain, K, rmin, rmax):
    print("Starting baseline initialization ...")
    # get some information
    Nu = Rtrain.shape[0]
    Ni = Rtrain.shape[1]
    
    # latent features matrices
    P = np.zeros((Nu, K))
    Q = np.zeros((K, Ni))
    
    # find average ratings value
    avgRatings = Rtrain.sum() / Rtrain.nnz
    
    # find users' biases
    usersBiases = []
    for u in range(Nu):
        nrate = Rtrain[u,:].nnz
        if (nrate == 0):
            uBias = avgRatings
        else:
            uBias = Rtrain[u,:].sum() / Rtrain[u,:].nnz
        usersBiases.append(uBias)
    usersBiases = usersBiases - avgRatings
    
    # find items' biases
    itemsBiases = []
    for i in range(Ni):
        nrate = Rtrain[:,i].nnz
        if (nrate == 0):
            iBias = avgRatings
        else:
            iBias = Rtrain[:,i].sum() / Rtrain[:,i].nnz
        itemsBiases.append(iBias)
    itemsBiases = itemsBiases - avgRatings
    
    # initialize P and Q
    # P
    tmp = avgRatings / (K - 2)
    P[:,0:(K - 2)] = tmp
    P[:,K-2] = usersBiases
    P[:,K-1] = 1
    
    # Q
    Q[0:(K - 1),:] = 1
    Q[K-1, :] = itemsBiases
    
    print("Finished baseline initialization!")
    return P, Q

def maxElement1(P, Q):
    maxE = 0
    # get P without first column and Q without first row
    P1 = P[:, 1:]
    Q1 = Q[1:, :]
    
    # find max element of P1.Q1
    maxRows = []
    Nu = P1.shape[0]
    Ni = Q1.shape[1]
    for u in range(Nu):
        Ru = P1[u,:].dot(Q1)
        maxRu = np.max(Ru)
        maxRows.append(maxRu)
    
    maxE = max(maxRows)
    return maxE

if __name__ == "__main__":
    nu = 5
    ni = 4
    k = 3
    rmin = 1
    rmax = 5
    R = sparse.dok_matrix((nu, ni), dtype=np.float32)
    R[0,0] = 1
    R[0,2] = 2
    R[1,1] = 4
    R[1,3] = 5
    R[2,0] = 3
    R[2,1] = 2
    R[3,0] = 1
    R[3,2] = 1
    R[4,1] = 2
    R[4,3] = 3
    P, Q = initParamsBaseline(R, k)
    print (P)
    print(Q)