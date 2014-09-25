import scipy.sparse as sparse
import numpy as np

def get_MovieLens10M_ExpData():
    # MovieLens 10M dataset
    usersNum = 71567
    itemsNum = 10681
    ratingsNum = 10000054
    attrNum = 19
    rmin = 1
    rmax = 5
    ratingsFile = "../data/ml-10M/ratings.dat"
    contentFile = "../data/ml-10M/movies.dat"
    
    # prepare items-attributes matrix
    print "Loading Items - attributes matrix ..."
    itemsDict = getContentMatrix(contentFile, itemsNum, attrNum)
    
    # Load the whole data
    print "Loading ratings matrix ..."
    R = getRatingsMatrix(ratingsFile, usersNum, itemsNum, itemsDict)
    
    return R, usersNum, itemsNum, rmin, rmax

def get_MovieLens_10M_Data(trainF, testF, contentF):
    # MovieLens 10M dataset
    usersNum = 71567
    itemsNum = 10681
    ratingsNum = 10000054
    attrNum = 19
    rmin = 1
    rmax = 5
    
    # prepare items-attributes matrix
    print "Loading Items - attributes matrix ..."
    itemsDict = getContentMatrix(contentF, itemsNum, attrNum)
    
    # prepare training ratings matrix
    print "Loading training ratings matrix ..."
    Rtrain = getRatingsMatrix(trainF, usersNum, itemsNum, itemsDict)
    
    # prepare test ratings matrix
    print "Loading test ratings matrix ..."
    Rtest = getRatingsMatrix(testF, usersNum, itemsNum, itemsDict)
    
    print "Finished preparing data!"
    return Rtrain, Rtest, usersNum, itemsNum, rmin, rmax

# OK
def getContentMatrix(fileName, Ni, Na):
    # initialization
#     A_T = np.zeros((Na, Ni))        # array Na x Ni
    itemsDict = {}                  # key: real item id, value: item order
#     contentDict = {"Action" : 0,
#                    "Adventure" : 1,
#                    "Animation" : 2,
#                    "Children" : 3,
#                    "Comedy" : 4,
#                    "Crime" : 5,
#                    "Documentary" : 6,
#                    "Drama" : 7,
#                    "Fantasy" : 8,
#                    "Film-Noir" : 9,
#                    "Horror" : 10,
#                    "IMAX" : 11,
#                    "Musical" : 12,
#                    "Mystery" : 13,
#                    "Romance" : 14,
#                    "Sci-Fi" : 15,
#                    "Thriller" : 16,
#                    "War" : 17,
#                    "Western" : 18}
    
    # read data from file
    ind = 0
    f = open(fileName)
    for line in f:
        elements = line.split("::")
        iid = int(elements[0])
#         genres = elements[2].split("|")
        
        # update item index dict
        itemsDict[iid] = ind
        
        # update items - attributes matrix
#         for genre in genres:
#             g = genre.rstrip('\n')
#             if (contentDict.has_key(g) == False):
#                 continue
#                 
#             A_T[contentDict[g], ind] = 1
        ind += 1
    
    f.close()
    return itemsDict
    
def getRatingsMatrix(fileName, Nu, Ni, iDict):
    # initialize matrix
    R = sparse.dok_matrix((Nu, Ni), dtype=np.float32)
    
    count = 0
    # read data from file
    f = open(fileName)
    for line in f:
        elements = line.split("::");
        uid = int(elements[0]) - 1
        iid = iDict[int(elements[1])]
        rate = float(elements[2])
        
        # assign values to the matrix
        R[uid, iid] = rate
        count += 1
    
    print "Successfully read %d ratings" % (count)
    f.close()
    return R

if __name__ == "__main__":
    trainingFile = "ml-10M/ra.train"
    testFile = "ml-10M/ra.test"
    contentFile = "ml-10M/movies.dat"
    get_MovieLens_10M_Data(trainingFile, testFile, contentFile)