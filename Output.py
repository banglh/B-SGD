import time
import numpy as np

def printToFile(methodName, ErrorType, K, trainErr, testErr, Info):
    # open file
    fileName = "results/" + methodName + "_K" + str(K) + "_" + ErrorType + "_" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".txt"
    f = open(fileName, 'w')
    
    # write results to file
    n = len(trainErr)
    f.write("%s\n" % (Info))
    f.write("=========================================================\n")
    f.write("%s\t%s\n" % ("Training", "Test"))
    for i in range(n-1):
        f.write("%f\t%f\n" % (trainErr[i + 1], testErr[i + 1]))
    
    print "Results have written to file %s" % (fileName)
    f.close()

# Ouput the result of a run of an experiment to file
def outputRunResults(fileName, trainErr, valErr, testErr, Info, moreResults):
    # open file
    f = open(fileName, 'w')
    
    # write some information to file
    f.write("%s\n" % (Info))
    f.write("=========================================================\n")
    
    # write results to file
    f.write("%s\t%s\n" % ("Training", "Validation"))
    n = len(trainErr)
    for i in range(n-1):
        f.write("%f\t%f\n" % (trainErr[i + 1], valErr[i + 1]))
    
    f.write("\nTest Error: %f" % (testErr))
    
    f.write("%s\n" % (moreResults))
    
    print "Results have written to file %s" % (fileName)
    f.close()

# Output the results of an experiment
def outputExpResults(fileName, testErrList, Info):
    # open file
    f = open(fileName, 'w')
    
    # write some information to file
    f.write("%s\n" % (Info))
    f.write("=========================================================\n")
    
    # write results to file
    i = 0
    for testErr in testErrList:
        i += 1
        f.write("\nRun %d - Test Error: %f" % (i, testErr))
    
    print "Results have written to file %s" % (fileName)
    f.close()
    
# Output the results of all experiments
def outputExpsResults(fileName, testErrList, Info):
    # open file
    f = open(fileName, 'w')
    
    # write some information to file
    f.write("%s\n" % (Info))
    f.write("=========================================================\n")
    
    # write results to file
    i = 0
    for testErr in testErrList:
        i += 1
        f.write("\nExperiment %d - (Average) Test Error: %f" % (i, testErr))
    
    f.write("\n Average Test Error: %f" % (np.mean(testErrList)))
    
    print "Results have written to file %s" % (fileName)
    f.close()
    
if __name__ == "__main__":
    methodName = "BMF"
    ErrorType = "RMSE"
    K = 10
    trainErr = [1.23, 3.23, 43.234, 34.234, 234.234]
    testErr = [1.23, 3.23, 43.234, 34.234, 234.234]
    printToFile(methodName, ErrorType, K, trainErr, testErr)
