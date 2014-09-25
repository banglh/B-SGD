import scipy.sparse as sparse
import numpy as np

def get_Jester_ExpData():
    # Jester dataset information
    R = 0
    nu = 73421
    ni = 100
    ratingsNum = 4136360
    rmin = -10
    rmax = 10
    dataFile1 = "../data/jester/jester-data-1.csv"
    dataFile2 = "../data/jester/jester-data-2.csv"
    dataFile3 = "../data/jester/jester-data-3.csv"
    
    # Initialize ratings matrix R
    R = sparse.dok_matrix((nu, ni), dtype=np.float32)
    startUser = 0
    
    # Read data files and fill in the matrix
    print "Reading data from file 1 ..."
    R, startUser, rNum1 = readDataFile(dataFile1, R, startUser)
    print "Reading data from file 2 ..."
    R, startUser, rNum2 = readDataFile(dataFile2, R, startUser)
    print "Reading data from file 3 ..."
    R, startUser, rNum3 = readDataFile(dataFile3, R, startUser)
    
    print "Finished preparing data (%d ratings)!" % (rNum1 + rNum2 + rNum3)
    
    return R, nu, ni, rmin, rmax

def readDataFile(fileName, R, startUser):
    endUser = startUser
    ratingsNum = 0
    usersNum = 0
    
    # open file
    f = open(fileName, "r")
    
    # read ratings data of each user
    for line in f:
        usersNum += 1
        elements = line.split(",");
        uid = endUser
        iid = 0
        
        # read user's ratings
        for rateStr in elements[1:]:
            rate = float(rateStr)
            if (rate != 99):
                R[uid,iid] = rate
                ratingsNum += 1
            iid += 1
        
        endUser += 1
    
    # print results
    print "Successfully read %d ratings of %d users from file %s" % (ratingsNum, usersNum, fileName)
    f.close()
    
    return R, endUser, ratingsNum

if __name__ == "__main__":
    R, nu, ni, rmin, rmax = get_Jester_ExpData()
    
    print "hehe"