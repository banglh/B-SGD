from DataPreparation import prepareExperimentData, getTrainValTest
from LatentFeaturesInit import initParamsRandom1
from Output import outputRunResults, outputExpResults, outputExpsResults
from BSGD_Engine import trainParamsExp
import numpy as np
import time
import os
import scipy.sparse as sparse
import cv2

if __name__ == "__main__":
    # params
#     trainPercents = [80, 60, 40, 20, 10]
    trainPercents = [80]
    valPercent = 5
    Klist = [10, 20, 50]
    version = 1
    muy = 1e-4
    alpha = 1
    rmin = 0
    rmax = 255
    
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
    
    
    print ("************* B-SGD EXPERIMENTS *************\n\n")

    # read image
    img = cv2.imread("lena.bmp")
    img = img[:,:,0]
    cv2.imwrite('original.bmp', img)

    # convert image to lower bitmap
    x = 4
    convertedImg = np.zeros((img.shape[0]/x, img.shape[1]/x))
    nrow = convertedImg.shape[0]
    ncol = convertedImg.shape[1]
    
    for row in range(nrow):
        for col in range(ncol):
            convertedImg[row,col] = np.sum(img[row*x:row*x+x, col*x:col*x+x]) / (x*x)
            
    cv2.imwrite('converted.bmp', convertedImg)
    
    for trainPercent in trainPercents:
        print("\n****************** Train: %d %% **********************\n" % (trainPercent))
        
        
        # prepare training, validation and test data
        Rtrain = sparse.dok_matrix(np.matrix(convertedImg))
        Rshape = Rtrain.shape
        Rval = sparse.dok_matrix((Rshape[0], Rshape[1]), dtype=np.float32)
        Rtest = sparse.dok_matrix((Rshape[0], Rshape[1]), dtype=np.float32) 
 
        testPercent = 100 - trainPercent - valPercent
        Rtrain, Rval, Rtest = getTrainValTest(Rtrain, Rtest, Rval, testPercent, valPercent)
        
        corruptedFileName = 'corrupted_' + str(trainPercent) + '.bmp'   
        cv2.imwrite(corruptedFileName , Rtrain.toarray())
        evalFileName = 'eval_' + str(trainPercent) + '.bmp'   
        cv2.imwrite(evalFileName , Rval.toarray())
        testFileName = 'test_' + str(trainPercent) + '.bmp'   
        cv2.imwrite(testFileName , Rtest.toarray())

    
        for K in Klist:
            print("\n****************** K: %d **********************\n" % (K))
            
            Info = "\nTraining data: " + str(trainPercent) + "%"
            Info += "\nValidation data: " + str(valPercent) + "%"
            Info += "\nTest data: " + str(testPercent) + "%"
            Info += "\n\nmuy = " + str(muy)
            Info += "\nalpha = " + str(alpha)
            Info += "\nnumber of latent features: " + str(K) + "\n"
            
            print (Info)

            # init latent feature matrices
            usersNum = Rtrain.shape[0]
            itemsNum = Rtrain.shape[1]
            P, Q = initParamsRandom1(usersNum, itemsNum, K, rmin, rmax)
            
            # train the feature matrices
            P, Q, trainErr, valErr, testErr = trainParamsExp(Rtrain, Rval, Rtest, P, Q, muy, alpha, rmin, rmax, version)
    
            # restore the original image
            restoredImg = Rtrain.toarray()
            estimatedImg = np.dot(P, Q)
            
            # TODO truncate pixel values in the estimated image
            overRangeList = []
            
            restoredImg[restoredImg == 0] = estimatedImg[restoredImg == 0]
            restoredFileName = 'restored_Train' + str(trainPercent) + '_K_' + str(K) + '.bmp'
            cv2.imwrite(restoredFileName, restoredImg)
            
            # write to file
            outFileName = baseFileName + str(trainPercent) + "_K_" + str(K) + ".txt"
            outputRunResults(outFileName, trainErr, valErr, testErr, Info)
    