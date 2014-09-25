from DataPreparation import prepareExperimentData, getTrainValTest
from LatentFeaturesInit import initParamsRandom1
from Output import outputRunResults, outputExpResults, outputExpsResults
from BSGD_Engine import trainParamsExp
import numpy as np
import time
import os
import scipy.sparse as sparse

if __name__ == "__main__":
    ############################### Main program ########################################
    # Dataset name
#     dataset = "MovieLens-10M"
#     dataset = "MovieLens-100k"
#     dataset = "Jester"
#     dataset = "Dating"
    dataset = "Book"
    
    # Proportion of training, validation and test set
    trainPercent = 85
    testPercent = 10
    valPercent = 5
    
    # BSGD version    # 1: rmin < PuQi < rmax with (u,i) in Rrated
                      # 2:  rmin < PuQi < rmax with (u,i) in R
    version = 1
    
    # Number of experiments running
    experimentsNumber = 3
    
    # List of values for number of latent features to test
    Klist = [50,10,20,5]
    
    # Hyper parameters
    muy = 1e-3
    alpha = 1
    
    # Information to write to the results files
    methodName = "BSGD-" + str(version)
    initMethod = "Random1"
    errorType = "RMSE"
    runningDate = time.strftime("%Y-%m-%d_%H-%M-%S")
    basePath = "results"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    path = "results/"
    baseFileName = path + runningDate + "_" + methodName + "-" + initMethod + "_" + errorType + "_"
    
    Info = "Dataset: " + dataset + "\n"
    Info += "\nTraining data: " + str(trainPercent) + "%"
    Info += "\nValidation data: " + str(valPercent) + "%"
    Info += "\nTest data: " + str(testPercent) + "%"
    Info += "\n\nmuy = " + str(muy)
    Info += "\nalpha = " + str(alpha)
    Info += "\n\nNumber of experiments: " + str(experimentsNumber)
    Info += "\nValues for number of latent features: " + str(Klist) + "\n"
    
    print ("************* B-SGD EXPERIMENTS *************\n\n")
    print (Info)
    
    # load ratings matrix
    Rtrain, usersNum, itemsNum, rmin, rmax = prepareExperimentData(dataset)
    Rshape = Rtrain.shape
    Rval = sparse.dok_matrix((Rshape[0], Rshape[1]), dtype=np.float32)
    Rtest = sparse.dok_matrix((Rshape[0], Rshape[1]), dtype=np.float32)
    
    # for each value of the number of latent features
    for K in Klist:
        print("\n****************** K = %d **********************\n" % (K))
        # results of all experiments
        expResults = []
        
        for exp in range(experimentsNumber):
            print("\n****************** K = %d - Experiment %d **********************\n" % (K, exp))
            # Randomly divide the data to Training, Validation and Test data
            Rtrain, Rval, Rtest = getTrainValTest(Rtrain, Rtest, Rval, testPercent, valPercent)
            
            # (For randome initialization) Run the model on the Training, Test and validation set made before
            repeatRuns = 5
            
            # results of all runs in the current experiment
            runResults = []
            
            for run in range(repeatRuns):
                print("\n****************** K = %d - Experiment %d - Run %d **********************\n" % (K, exp, run))
                # Initialize latent features matrices (P: Nu x k, B: k x Na)
                P, Q = initParamsRandom1(usersNum, itemsNum, K, rmin, rmax)
                
                # Training latent features matrices
                P, Q, trainErr, valErr, testErr = trainParamsExp(Rtrain, Rval, Rtest, P, Q, muy, alpha, rmin, rmax, version)
                
                # Record the results of the run to file
                runFileName = baseFileName + "K" + str(K) + "_" + "Exp-" + str(exp) + "_" + "Run-" + str(run) + ".txt"
                runInfo = Info
                outputRunResults(runFileName, trainErr, valErr, testErr, runInfo)
                
                # Record the result on test set to the list
                runResults.append(testErr)
            
            # Record the results of the experiment to file
            expFileName = baseFileName + "K" + str(K) + "_" + "Exp-" + str(exp) + ".txt"
            expInfo = Info
            outputExpResults(expFileName, runResults, expInfo)
            
            # Record the result of the current experiment
            expResults.append(np.mean(runResults))
            
        # Record the results of all experiments to file
        KfileName = baseFileName + "K" + str(K) + ".txt"
        KInfo = Info
        outputExpsResults(KfileName, expResults, KInfo)