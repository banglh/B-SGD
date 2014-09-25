import scipy.sparse as sparse
import numpy as np

def get_MovieLens100k_ExpData():
    # MovieLens 100k dataset
    usersNum = 943
    itemsNum = 1682
    ratingsNum = 100000
    rmin = 1
    rmax = 5
    ratingsFile = "../data/ml-100k/u.data"
    
    # Load the whole data
    print "Loading ratings matrix ..."
    R = getRatingsMatrix(ratingsFile, usersNum, itemsNum)

    return R, usersNum, itemsNum, rmin, rmax

def get_MovieLens_100K_Data(trainF, testF):
    # MovieLens 100k dataset
    usersNum = 943
    itemsNum = 1682
    ratingsNum = 100000
    rmin = 1
    rmax = 5
    
    # prepare training ratings matrix
    print "Loading training ratings matrix ..."
    Rtrain = getRatingsMatrix(trainF, usersNum, itemsNum)
    
    # prepare test ratings matrix
    print "Loading test ratings matrix ..."
    Rtest = getRatingsMatrix(testF, usersNum, itemsNum)
    
    print "Finished preparing data!"
    return Rtrain, Rtest, usersNum, itemsNum, rmin, rmax

def getRatingsMatrix(fileName, Nu, Ni):
    # initialize matrix
    R = sparse.dok_matrix((Nu, Ni), dtype=np.float32)
    
    count = 0
    # read data from file
    f = open(fileName)
    for line in f:
        elements = line.split("\t");
        uid = int(elements[0]) - 1
        iid = int(elements[1]) - 1
        rate = float(elements[2])
        
        # assign values to the matrix
        R[uid, iid] = rate
        count += 1
    
    print "Successfully read %d ratings" % (count)
    f.close()
    return R

if __name__ == "__main__":
    trainingFile = "ml-100k/ua.base"
    testFile = "ml-100k/ua.test"
    contentFile = "ml-100k/u.item"
    get_MovieLens_100K_Data(trainingFile, testFile, contentFile)