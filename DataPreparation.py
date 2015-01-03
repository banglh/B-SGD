from datasets.MovieLens_10M import get_MovieLens_10M_Data, get_MovieLens10M_ExpData
from datasets.MovieLens_100k import get_MovieLens_100K_Data, get_MovieLens100k_ExpData
from datasets.Book import get_Book_ExpData
from datasets.Jester import get_Jester_ExpData
from datasets.Dating import get_Dating_ExpData
import random 

def prepareData(dataset, trainF, testF, contentF):
    # outputs
    Rtrain = 0
    Rtest = 0
    nu = 0
    ni = 0
    
    if (dataset == "MovieLens-10M"):
        Rtrain, Rtest, nu, ni, rmin, rmax = get_MovieLens_10M_Data(trainF, testF, contentF)
    elif (dataset == "MovieLens-100K"):
        Rtrain, Rtest, nu, ni, rmin, rmax = get_MovieLens_100K_Data(trainF, testF)
        
    return Rtrain, Rtest, nu, ni, rmin, rmax

def prepareExperimentData(datasetName):
    # outputs
    R = 0
    nu = 0
    ni = 0
    
    if (datasetName == "MovieLens-10M"):
        R, nu, ni, rmin, rmax = get_MovieLens10M_ExpData()
    elif (datasetName == "Jester"):
        R, nu, ni, rmin, rmax = get_Jester_ExpData()
    elif (datasetName == "Dating"):
        R, nu, ni, rmin, rmax = get_Dating_ExpData()
    elif (datasetName == "Book"):
        R, nu, ni, rmin, rmax = get_Book_ExpData()
    elif (datasetName == "MovieLens-100k"):
        R, nu, ni, rmin, rmax = get_MovieLens100k_ExpData()
    return R, nu, ni, rmin, rmax

def getTrainValTest(Rtrain, Rtest, Rval, testPercent, valPercent):
    # recover the full ratings matrix
    for key in Rtest.iterkeys():
        u = key[0]
        i = key[1]
        Rtrain[u,i] = Rtest[u,i]
#         Rtest[u,i] = 0.0            --> error: change size of dictionary during iteration
    Rtest.clear()
    
    for key in Rval.iterkeys():
        u = key[0]
        i = key[1]
        Rtrain[u,i] = Rval[u,i]
#         Rval[u,i] = 0.0               --> error: change size of dictionary during iteration
    Rval.clear()
    
    # calculate number of observations in Test and validation set
    nObs = Rtrain.getnnz()
    nTestObs = int(testPercent * nObs / 100)
    nValObs = int(valPercent * nObs / 100)
    
    # prepare test set
    nonzeroIndices = Rtrain.nonzero()
    nonzeroUserIndex = nonzeroIndices[0]
    nonzeroItemIndex = nonzeroIndices[1]
    testPositions = random.sample(xrange(nObs), nTestObs)
    print "nObs: %d, nTestObs: %d" % (nObs, nTestObs)
    for pos in testPositions:
        u = nonzeroUserIndex[pos]
        i = nonzeroItemIndex[pos]
        
        # move the observation to test set
        Rtest[u,i] = Rtrain[u,i]
        Rtrain[u,i] = 0.0
    
    # prepare validation set
    nonzeroIndices = Rtrain.nonzero()
    nonzeroUserIndex = nonzeroIndices[0]
    nonzeroItemIndex = nonzeroIndices[1]
    validationPositions = random.sample(xrange(len(nonzeroUserIndex)), nValObs)
    for pos in validationPositions:
        u = nonzeroUserIndex[pos]
        i = nonzeroItemIndex[pos]
        
        # move the observation to validation set
        Rval[u,i] = Rtrain[u,i]
        Rtrain[u,i] = 0.0
    
    # print resulst
    print("Finished preparing Training data (%d ratings), Test data (%d ratings) and Validation data(%d ratings)\n" % (nObs - nTestObs - nValObs, nTestObs, nValObs))
    return Rtrain, Rval, Rtest