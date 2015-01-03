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
    trainPercents = [80, 60, 40, 20, 10]
#     trainPercents = [80]
    valPercent = 5
    Klist = [10, 20, 50]
#     Klist = [10]
    version = 1
    muy = 1e-4
    alpha = 1
    lamda = 1
    rmin = 1
    rmax = 256
    
    # Information to write to the results files
    methodName = "BSGD-" + str(version)
    initMethod = "Random1"
    errorType = "RMSE"
    runningDate = time.strftime("%Y-%m-%d_%H-%M-%S")
    basePath = "results" + runningDate + "/"
    if not os.path.exists(basePath):
        os.makedirs(basePath)
    baseFileName = basePath + runningDate + "_" + methodName + "-" + initMethod + "_" + errorType + "_"
    
    
    print ("************* B-SGD EXPERIMENTS *************\n\n")

    # read image
    img = cv2.imread("lena4.bmp")
    img = img[:,:,0]
    img = img.astype(np.float64)
    img += 1    # avoid pixel with value equals to zero (range: 0~255 --> range 1~256)

    for trainPercent in trainPercents:
        print("\n****************** Train: %d %% **********************\n" % (trainPercent))
        
        # prepare training, validation and test data
        Rtrain = sparse.dok_matrix(np.matrix(img))
        Rshape = Rtrain.shape
        Rval = sparse.dok_matrix((Rshape[0], Rshape[1]), dtype=np.float64)
        Rtest = sparse.dok_matrix((Rshape[0], Rshape[1]), dtype=np.float64) 
 
        testPercent = 100 - trainPercent - valPercent
        Rtrain, Rval, Rtest = getTrainValTest(Rtrain, Rtest, Rval, testPercent, valPercent)
        
        corruptedFileName = basePath + 'train' + str(trainPercent) + '.bmp'   
        cv2.imwrite(corruptedFileName , Rtrain.toarray())
        evalFileName = basePath + 'validation' + str(trainPercent) + '.bmp'   
        cv2.imwrite(evalFileName , Rval.toarray())
        testFileName = basePath + 'test_' + str(trainPercent) + '.bmp'   
        cv2.imwrite(testFileName , Rtest.toarray())

    
        for K in Klist:
            print("\n****************** K: %d **********************\n" % (K))
            
            Info = "\nTraining data: " + str(trainPercent) + "%"
            Info += "\nValidation data: " + str(valPercent) + "%"
            Info += "\nTest data: " + str(testPercent) + "%"
            Info += "\nMethod name: " + methodName
            Info += "\nInit method: " + initMethod
            Info += "\n\nmuy = " + str(muy)
            Info += "\nalpha = " + str(alpha)
            Info += "\nlambda = " + str(lamda)
            Info += "\nrmin = " + str(rmin)
            Info += "\nrmax = " + str(rmax)
            Info += "\nnumber of latent features: " + str(K) + "\n"
            
            print (Info)

            # init latent feature matrices
            usersNum = Rtrain.shape[0]
            itemsNum = Rtrain.shape[1]
            P, Q = initParamsRandom1(usersNum, itemsNum, K, rmin, rmax)
            
            # train the feature matrices
            P, Q, trainErr, valErr, testErr = trainParamsExp(Rtrain, Rval, Rtest, P, Q, muy, alpha, lamda, rmin, rmax, version)
    
            # restore the original image (training + validation + estimated values on test set (with truncation))
            restoredImg = Rtrain.toarray()
            RvalArr = Rval.toarray()
            restoredImg[restoredImg == 0] = RvalArr[restoredImg == 0]   # fill in the pixels from Rval
            
            estimatedImg = np.dot(P, Q)
            restoredImg[restoredImg == 0] = estimatedImg[restoredImg == 0]  # fill in the test pixels from estimatedImg
            
            restoredImg[restoredImg > rmax] = rmax  # truncate pixels values
            restoredImg[restoredImg < rmin] = rmin  # truncate pixels values
            
            restoredImg = np.round(restoredImg)
            restoredImg -= 1    # convert to 0~255
            restoredImg = restoredImg.astype(int)
            
            restoredFileName = basePath + 'restored_Train' + str(trainPercent) + '_K_' + str(K) + '.bmp'
            cv2.imwrite(restoredFileName, restoredImg)
            
            # check estimated pixel values not in range
            RtestArr = Rtest.toarray()
            RtrainArr = Rtrain.toarray()

            overRange = np.where(estimatedImg > rmax)
            underRange = np.where(estimatedImg < rmin)
            outRangeTrain = estimatedImg[((estimatedImg < rmin) | (estimatedImg > rmax)) & (RtrainArr != 0)]
            outRangeVal = estimatedImg[((estimatedImg < rmin) | (estimatedImg > rmax)) & (RvalArr != 0)]
            outRangeTest = estimatedImg[((estimatedImg < rmin) | (estimatedImg > rmax)) & (RtestArr != 0)]
            
            estimatedImg = np.round(estimatedImg)
            estimatedImg -= 1
            estimatedImg = estimatedImg.astype(int)
            checkRangeImg = np.zeros((Rshape[0], Rshape[1], 3))
            checkRangeImg[:,:,0] = estimatedImg
            checkRangeImg[:,:,1] = estimatedImg
            checkRangeImg[:,:,2] = estimatedImg
            
            nOver = len(overRange[0])
            nUnder = len(underRange[0])
            
            # fill over range pixels with red color
            for i in range(nOver):
                checkRangeImg[overRange[0][i], overRange[1][i], 0] = 0
                checkRangeImg[overRange[0][i], overRange[1][i], 1] = 0
                checkRangeImg[overRange[0][i], overRange[1][i], 2] = 255

            # fill under range pixels with green color
            for i in range(nUnder):
                checkRangeImg[underRange[0][i], underRange[1][i], 0] = 0
                checkRangeImg[underRange[0][i], underRange[1][i], 1] = 255
                checkRangeImg[underRange[0][i], underRange[1][i], 2] = 0
            
            estimateFileName = basePath + 'estimated_Train' + str(trainPercent) + '_K_' + str(K) + '.bmp'
            cv2.imwrite(estimateFileName, checkRangeImg)

            # calculate proportion of out-of-range pixes
            outRangeRate = np.double((len(overRange[0]) + len(underRange[0]))) / (usersNum * itemsNum) * 100
            outRangeTrainRate = np.double(len(outRangeTrain)) / Rtrain.nnz * 100
            outRangeValRate = np.double(len(outRangeVal)) / Rval.nnz * 100
            outRangeTestRate = np.double(len(outRangeTest)) / Rtest.nnz * 100
            
            moreResults = "\n"
            moreResults += "\nOut of range rate: " + str(outRangeRate) + "%"
            moreResults += "\nOut of range rate (train): " + str(outRangeTrainRate) + "%"
            moreResults += "\nOut of range rate (validation): " + str(outRangeValRate) + "%"
            moreResults += "\nOut of range rate (test): " + str(outRangeTestRate) + "%"
            
            # write to file
            outFileName = baseFileName + str(trainPercent) + "_K_" + str(K) + ".txt"
            outputRunResults(outFileName, trainErr, valErr, testErr, Info, moreResults)
    