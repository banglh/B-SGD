from os import listdir
from os.path import isfile, join
import numpy as np

def getResults(directoryPath, nExp, nRun):
    # get list of files in current directory
    files = listdir(directoryPath)
    
    # array of results
    results = np.zeros((nRun, nExp))
    
    for f in files:
        # get full path
        path = directoryPath + "/" + f
        
        elements = str.split(f, "_")
        runStr = str.split(elements[7], ".")[0]
        expStr = elements[6]
        
        run = int(runStr[3:])
        exp = int(expStr[3:])
        
        # open file
        fObj = open(path, "r")
        for line in fObj:
            lineElements = str.split(line, ":")
            if (lineElements[0] == "Test RMSE"):
                results[run,exp] = float(lineElements[-1])
        
        fObj.close()
    
    # print results
    for run in range(nRun):
        runStr = ""
        for exp in range(nExp-1):
            runStr += str(results[run,exp]) + "\t"
        runStr += str(results[run, nExp-1])
        print runStr
    
    print ""
    
if __name__ == "__main__":
    directoryPath1 = "C:/Users/MrICE/Dropbox/research/B-SGD results/BSGD-1-boundedrandom-1/Dating/bounded-random/K10"
    directoryPath2 = "C:/Users/MrICE/Dropbox/research/B-SGD results/BSGD-1-boundedrandom-1/Dating/bounded-random/K20"
    directoryPath3 = "C:/Users/MrICE/Dropbox/research/B-SGD results/BSGD-1-boundedrandom-1/Dating/bounded-random/K50"
    getResults(directoryPath1, 5, 5)
    getResults(directoryPath2, 5, 5)
    getResults(directoryPath3, 5, 5)