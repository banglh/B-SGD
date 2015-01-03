import sys
from Error import RMSE
import numpy as np
import time

def trainParamsExp(Rtrain, Rval, Rtest, P0, Q0, muy, alpha, lamda, rmin, rmax, version):
    maxIteration = 200
    P = P0
    Q = Q0
    Plast = 0.0
    Qlast = 0.0
    
    # get known params
    nu = Rtrain.shape[0]        # number of users
    ni = Rtrain.shape[1]        # number of items
    K = P0.shape[1]             # number of latent features
    
    train_Err = []
    val_Err = []
    
    train_Err.append(sys.float_info.max)
    val_Err.append(sys.float_info.max)
    
    # calculate initial RMSE
    train_err = RMSE(Rtrain, P, Q)
    val_err = RMSE(Rval, P, Q)
    train_Err.append(train_err)
    val_Err.append(val_err)
    
    print "Initial RMSE:    Training set: %f    -    Validation set: %f" % (train_err, val_err)
    
    iteration = 0
    
    # calculate some constants
    exp_alpha_rmax = np.exp(alpha * rmax)
    exp_alpha_rmin = np.exp(alpha * rmin)
    
    iterNum = 0
    while (iterNum < maxIteration and not(stopCriterion(val_Err[-1], val_Err[-2]))):
        iterNum += 1
        start = time.time()
        
        iteration += 1
        Plast = P.copy()
        Qlast = Q.copy()
        
        # update parameters
        for key in Rtrain.iterkeys():
            u = key[0]
            i = key[1]
            
            # current estimated Rui
            estimated_Rui = P[u,:].dot(Q[:,i])
            if (estimated_Rui > 790 or estimated_Rui < -745):
                print "erui: %f\n" % (estimated_Rui)
            # current error
            Eui = Rtrain[u,i] - estimated_Rui
            
            if (version == 2):  #(version 2)
                # calculate QHi
                QHi = 0.0
                for item in range(ni):
                    e_rui = P[u,:].dot(Q[:,item])
                    Hi = np.exp(alpha * (e_rui - rmax)) - np.exp(alpha * (rmin - e_rui))
                    QHi += Q[:,item] * Hi
                
                # calculate PHu
                PHu = 0.0
                for user in range(nu):
                    e_rui = P[user,:].dot(Q[:,i])
                    Hu = np.exp(alpha * (e_rui - rmax)) - np.exp(alpha * (rmin - e_rui))
                    PHu = P[user,:] * Hu
                    
                # update Pu, Qi
                P[u,:] = P[u,:] + muy * (Eui * Q[:,i] - lamda * alpha * QHi)
                Q[:,i] = Q[:,i] + muy * (Eui * P[u,:] - lamda * alpha * PHu)
                
            else: #(version 1)
#                 H = np.exp(alpha * (estimated_Rui - rmax)) - np.exp(alpha * (rmin - estimated_Rui))
                exp_alpha_rui = np.exp(alpha * estimated_Rui)
                H = (exp_alpha_rui / exp_alpha_rmax) - (exp_alpha_rmin / exp_alpha_rui)
                
                # update Pu, Qi
                tmp = muy * (Eui - lamda * alpha * H)
                P[u,:] = P[u,:] + tmp * Q[:,i]
                Q[:,i] = Q[:,i] + tmp * P[u,:]
        
        # calculate RMSE and print the results
        train_err = RMSE(Rtrain, P, Q)
        val_err = RMSE(Rval, P, Q)
        train_Err.append(train_err)
        val_Err.append(val_err)
        
        elapsed = (time.time() - start) / 60
        print "RMSEs at iteration %d:    Training set: %f    -    Validation set: %f    -    Updating time: %f" % (iteration, train_err, val_err, elapsed)
    
    test_Err =  RMSE(Rtest, Plast, Qlast)
    print "\nFinished training, RMSE on test data: %f\n" % (test_Err)
    
    return P, Q, train_Err, val_Err, test_Err

# stop criterion
def stopCriterion(curRMSE, lastRMSE):
    stop = False
    threshold = 1e-4
    
    # if RMSE start increasing or the change of RMSE is too small
    if ((curRMSE > lastRMSE) or ((lastRMSE - curRMSE) < threshold)):
        stop = True
        
    return stop