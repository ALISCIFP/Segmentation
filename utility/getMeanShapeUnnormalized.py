# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:51:35 2016

@author: Kangyan Zhou
"""

import numpy as np

fileName = "list_landmarks_align_celeba.txt"

#read file
def readAnnotationFile():
    with open(fileName, "r") as file:
        allLandmarks = []
        totalLeftEyeX = 0
        totalLeftEyeY = 0
        totalRightEyeX = 0
        totalRightEyeY = 0
        totalLeftMouthX = 0
        totalLeftMouthY = 0
        totalRightMouthX = 0
        totalRightMouthY = 0
        totalNoseX = 0
        totalNoseY = 0
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
            
            totalLeftEyeX += leftEyeX
            totalLeftEyeY += leftEyeY
            totalRightEyeX += rightEyeX
            totalRightEyeY += rightEyeY
            totalNoseX += noseX
            totalNoseY += noseY
            totalLeftMouthX += leftMouthX
            totalLeftMouthY += leftMouthY
            totalRightMouthX += rightMouthX
            totalRightMouthY += rightMouthY        

        totalLeftEyeX /= i-2
        totalLeftEyeY /= i-2
        totalRightEyeX /= i-2
        totalRightEyeY /= i-2
        totalNoseX /= i-2
        totalNoseY /= i-2
        totalLeftMouthX /= i-2
        totalLeftMouthY /= i-2
        totalRightMouthX /= i-2
        totalRightMouthY /= i-2 
        
        leftEye = np.array([totalLeftEyeX, totalLeftEyeY])
        rightEye = np.array([totalRightEyeX, totalRightEyeY])
        nose = np.array([totalNoseX, totalNoseY])
        leftMouth = np.array([totalLeftMouthX, totalLeftMouthY])
        rightMouth = np.array([totalRightMouthX, totalRightMouthY])
        
        meanShape = [leftEye, rightEye, nose, leftMouth, rightMouth]
        np.savetxt("meanShape.txt", meanShape)
       
        return
    
if __name__ == "__main__":
    readAnnotationFile()
