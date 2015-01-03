import numpy as np

def RMSE(R, P, Q):
    rmse = 0.0
    
    for key in R.iterkeys():
        j = key[0]
        i = key[1]
        
        # estimated R[j,i]
        eRji = P[j,:].dot(Q[:,i])
        
        # update rmse
        delta = eRji - R[j,i]
        rmse += delta * delta
    
    rmse = np.sqrt(rmse / R.nnz)
    return rmse