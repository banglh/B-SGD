import scipy.sparse as sparse
import numpy as np

def get_Dating_ExpData():
    nu = 135359
    ni = 168791
    ratingsNum = 17359346
    rmin = 1
    rmax = 10
    ratingsFile = "../data/dating/ratings.dat"
    
    # Initialize ratings matrix R
    R = sparse.dok_matrix((nu, ni), dtype=np.float32)
    
    # Load the whole data
    print "Loading ratings matrix ..."
    R = getRatingsMatrix(ratingsFile, R)
    
    return R, nu, ni, rmin, rmax

def getRatingsMatrix(fileName, R):
    count = 0
    itemsDict = {}                  # key: real item id, value: item order
    ind = 0
    
    # read data from file
    f = open(fileName)
    for line in f:
        elements = line.split(",");
        uid = int(elements[0]) - 1
        itemID = elements[1]
        if (not(itemsDict.has_key(itemID))):
            itemsDict[itemID] = ind
            ind += 1
        iid = itemsDict[itemID]
        rate = float(elements[2])
        
        # assign values to the matrix
        if (rate != 0.0):
            R[uid, iid] = rate
            count += 1
    
    print "Successfully read %d ratings on %d items" % (count, ind)
    f.close()
    
    return R

if __name__ == "__main__":
    R, nu, ni, rmin, rmax = get_Dating_ExpData()