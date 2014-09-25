import scipy.sparse as sparse
import numpy as np

def get_Book_ExpData():
    # Book Crossing dataset
    nu = 278858
    ni = 271379
    ratingsNum = 1149780
    rmin = 1
    rmax = 10
    bookFile = "../data/book/BX-Books.csv"
    userFile = "../data/book/BX-Users.csv"
    ratingsFile = "../data/book/BX-Book-Ratings.csv"
    
    # Initialize ratings matrix R
    R = sparse.dok_matrix((nu, ni), dtype=np.float32)
    
    # Read information about books
    print "Loading books information ..."
    itemsDict = getBooksInfo(bookFile)
    
    # Load the whole data
    print "Loading ratings matrix ..."
    R = getRatingsMatrix(ratingsFile, R, itemsDict)
    
    return R, nu, ni, rmin, rmax

def getBooksInfo(fileName):
    # initialization
    itemsDict = {}                  # key: real item id, value: item order
    
    # read data from file
    ind = 0
    f = open(fileName)
    f.readline()
    for line in f:
        elements = line.split(";")
        iid = elements[0].replace('"', '')
        
        # update item index dict
        itemsDict[iid] = ind
        
        ind += 1
    
    f.close()
    return itemsDict

# current ERROR:
# 1. book with ISSBN "3257224281" is not contained in books information file
def getRatingsMatrix(fileName, R, iDict):
    count = 0
    unknownCount = 0
    
    # read data from file
    f = open(fileName)
    f.readline()
    for line in f:
        elements = line.split(";");
        uid = int(elements[0].replace('"', '')) - 1
        isbn = elements[1].replace('"', '')
        if (iDict.has_key(isbn)):
            iid = iDict[isbn]
            rate = float(elements[2].replace('"', ''))
            
            # assign values to the matrix
            if (rate != 0.0):
                R[uid, iid] = rate
                count += 1
        else:
            unknownCount += 1
    
    print "Successfully read %d ratings" % (count)
    print "Number of unknown items: %d" % (unknownCount)
    f.close()
    return R

def getRatingsMatrix2(fileName, R):
    count = 0
    itemsDict = {}                  # key: real item id, value: item order
    ind = 0
    
    # read data from file
    f = open(fileName)
    f.readline()
    for line in f:
        elements = line.split(";");
        uid = int(elements[0].replace('"', '')) - 1
        isbn = elements[1].replace('"', '')
        if (not(itemsDict.has_key(isbn))):
            itemsDict[isbn] = ind
            ind += 1
        iid = itemsDict[isbn]
        rate = float(elements[2].replace('"', ''))
        
        # assign values to the matrix
        if (rate != 0.0):
            print "iid = %d" % (iid)
            R[uid, iid] = rate
            count += 1
    
    print "Successfully read %d ratings" % (count)
    f.close()
    
    return R

if __name__ == "__main__":
    R, nu, ni, rmin, rmax = get_Book_ExpData()