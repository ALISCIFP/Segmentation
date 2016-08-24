# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:27:46 2016

@author: Kangyan Zhou
"""

import os  
import re
import numpy as np
import h5py
from scipy.spatial import procrustes

fileName = "list_landmarks_align_celeba.txt"

#read file
def readAnnotationFile():
    with open(fileName, "r") as file:
        allLandmarks = []
        for i, line in enumerate(file):
            line = line.split()
            if(i == 1 or i == 0):
                continue
            
            leftEyeX = np.float(line[1])
            leftEyeY = np.float(line[2])
            
            rightEyeX = np.float(line[3])
            rightEyeY = np.float(line[4])
            
            noseX = np.float(line[5])
            noseY = np.float(line[6])
            
            leftMouthX = np.float(line[7])
            leftMouthY = np.float(line[8])
            
            rightMouthX = np.float(line[9])
            rightMouthY = np.float(line[10].replace("\n", ""))
        
#            allX = [leftEyeX, rightEyeX, noseX, leftMouthX, rightMouthX]
#            allY = [leftEyeY, rightEyeY, noseY, leftMouthY, rightMouthY]
#        
#            meanX = np.mean(allX)
#            meanY = np.mean(allY)
#            
#            leftEyeX, rightEyeX, noseX, leftMouthX, rightMouthX = allX - meanX
#            leftEyeY, rightEyeY, noseY, leftMouthY, rightMouthY = allY - meanY
            
            leftEye = np.array([leftEyeX, leftEyeY])
            rightEye = np.array([rightEyeX, rightEyeY])
            nose = np.array([noseX, noseY])
            leftMouth = np.array([leftMouthX, leftMouthY])
            rightMouth = np.array([rightMouthX, rightMouthY])
            
            allLandmarksPerPicture = [leftEye, rightEye, nose, leftMouth, rightMouth]
            allLandmarks.append(allLandmarksPerPicture)
            
        allLandmarks = np.array(allLandmarks)
        return allLandmarks

def procrustes2d(origin):
    f = np.copy(origin)
    numberOfShapes = np.shape(f)[0]
    
    diff = np.inf
    timesToConverge = 0        
    while(timesToConverge < 2):
        print(timesToConverge)
        if(timesToConverge == 0):
            pivotIndex = np.random.randint(0, numberOfShapes)
            for i in range(0, numberOfShapes):
                if(i == pivotIndex):
                    continue
                mtx1, mtx2, disparity = procrustes(f[pivotIndex], f[i])
                f[pivotIndex] = mtx1
                f[i] = mtx2
        else:
            for i in range(0, numberOfShapes):
                mtx1, mtx2, disparity = procrustes(newMean[0], f[i])
                f[i] = mtx2
                
        newMean = np.sum(f, axis = 0, keepdims=True)/numberOfShapes
        diff = np.sqrt(np.sum(np.square(f-newMean)))
        print(diff)
        timesToConverge += 1
    
    print(diff)
    return f, newMean

if __name__ == "__main__":
    tmp = readAnnotationFile()
    normalizedShapes, meanShape = procrustes2d(tmp)
    np.savetxt("meanShape2.txt", meanShape[0])
    np.save("normalizedShapes", normalizedShapes)
            
