# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import deepdish as dd
from scipy import misc
from itertools import product
import os

imgDir = '/home/alisc/data/img_align_celeba'

def bilinearInterpolation(x, y, points):
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def getTransCoord(x, y, transMatrix):
    matrix2 = np.array([x, y, 1]).T
    
    coord = transMatrix.T * matrix2
    
    return coord[0], coord[1]

def readImgGetTrans(filename, transMatrix):
    img = misc.imread(filename)
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    
    redChannel = []
    greenChannel = []
    blueChannel = []
    for x, y in product(range(len(red)), repeat = 2):
        x_, y_ = getTransCoord(x, y, transMatrix)
        smallX = int(x_)
        smallY = int(y_)
        largeX = int(x_) + 1
        largeY = int(y_) + 1 
        redValue = bilinearInterpolation(x_, y_, tuple(smallX, smallY, red[smallX][smallY]), \
                                            tuple(smallX, largeY, red[smallX][largeY]), \
                                            tuple(largeX, smallY, red[largeX][smallY]), \
                                            tuple(largeX, largeY, red[largeX][largeY]))
        
        greenValue = bilinearInterpolation(x_, y_, tuple(smallX, smallY, green[smallX][smallY]), \
                                            tuple(smallX, largeY, green[smallX][largeY]), \
                                            tuple(largeX, smallY, green[largeX][smallY]), \
                                            tuple(largeX, largeY, green[largeX][largeY]))
        
        
        blueValue = bilinearInterpolation(x_, y_, tuple(smallX, smallY, blue[smallX][smallY]), \
                                            tuple(smallX, largeY, blue[smallX][largeY]), \
                                            tuple(largeX, smallY, blue[largeX][smallY]), \
                                            tuple(largeX, largeY, blue[largeX][largeY]))
    
        redChannel.append(redValue)
        greenChannel.append(greenValue)
        blueChannel.append(blueValue)
    
    return [np.reshape(redChannel, (24, 24)), np.reshape(greenChannel, (24, 24)), np.reshape(blueChannel, (24, 24))]
        

if __name__ == '__main__':
    transMatrix = dd.io.load('test.h5')
    result = []
    pwd = os.getcwd()
    os.chdir(imgDir)
    for i in os.listdir(imgDir):
        if i.endswith(".jpg"): 
            filename = os.path.basename(i)
            print(filename)
            key = int(filename.split('.')[0])
            transImg = readImgGetTrans(i, transMatrix[key])
            result.append(transImg)
            break
        else:
            continue
    
    result = np.array(result)    
    os.chdir(pwd)
    np.save('meanShape24_24', np.mean(result, axis = 0))
