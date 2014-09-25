from DataPreparation import prepareData
from LatentFeaturesInit import initParamsRandom1
from Output import printToFile
from BSGD_Engine import trainParams

if __name__ == "__main__":
    ############################### Dataset information #################################
#     MovieLens 10M dataset
#     dataset = "MovieLens-10M"
#     trainingFile = "../data/ml-10M/r1.train"
#     testFile = "../data/ml-10M/r1.test"
#     contentFile = "../data/ml-10M/movies.dat"
    
    # MovieLens 10M dataset
    dataset = "MovieLens-100K"
    trainingFile = "../data/ml-100k/ua.base"
    testFile = "../data/ml-100k/ua.test"
    contentFile = "../data/ml-100k/u.item"
    
    ############################### Main program ########################################
    
    # prepare training data and test data
    Rtrain, Rtest, usersNum, itemsNum, rmin, rmax = prepareData(dataset, trainingFile, testFile, contentFile)
    
    version = 2         # 1: rmin < PuQi < rmax with (u,i) in Rrated
                        # 2:  rmin < PuQi < rmax with (u,i) in R
    Klist = [5,10,20,50]
    muy = 10e-4
    alpha = 1
    
    Info = "Dataset: " + dataset + "\n"
    Info += "training file: " + trainingFile + "\n"
    Info += "testing file: " + testFile + "\n"
    Info += "content file: " + contentFile + "\n"
    Info += "muy = " + str(muy) + "\n"
    Info += "alpha = " + str(alpha) + "\n"
    
    for K in Klist:
        
        # Initialize latent features matrices (P: Nu x k, B: k x Na)
        P, Q = initParamsRandom1(usersNum, itemsNum, K, rmin, rmax)
        
        # Training latent features matrices
        P, Q, trainRMSE, testRMSE = trainParams(Rtrain, Rtest, P, Q, muy, alpha, rmin, rmax, version)
        
        # Ouput results to file
        if (version == 2):
            printToFile("B-SGD-2-Random", "RMSE", K, trainRMSE, testRMSE, Info)
        else:
            printToFile("B-SGD-1-Random", "RMSE", K, trainRMSE, testRMSE, Info)