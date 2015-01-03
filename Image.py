import numpy as np
import time
import os
import scipy.sparse as sparse
import cv2

def convertImg():
    # read image
    img = cv2.imread("lena.bmp")
    img = img[:,:,0]
    cv2.imshow("lena", img)
    xList = [2,3,4]
    
    for x in xList:
        convertedImg = np.zeros((img.shape[0]/x, img.shape[1]/x))
        nrow = convertedImg.shape[0]
        ncol = convertedImg.shape[1]
    
        for row in range(nrow):
            for col in range(ncol):
                convertedImg[row,col] = np.double(np.sum(img[row*x:row*x+x, col*x:col*x+x])) / (x*x)
            
        convertedImg = np.round(convertedImg)
        convertedImg = convertedImg.astype(int)
        
        fileName = "lena" + str(x) + ".bmp"
        cv2.imwrite(fileName, convertedImg)
    
    cv2.waitKey(0)
    
if __name__ == "__main__":
    img = cv2.imread("lena.bmp")
    img[:,:,0] = 256
    a = np.zeros((600,600,3))
    a [:,:,0] = 300
    cv2.imshow("lena", img)
    cv2.waitKey(0)