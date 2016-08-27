# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import numpy as np
import deepdish as dd
from scipy import misc
from itertools import product
import os
from multiprocessing import Pool

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
    matrix2 = np.array([x, y, 1])
    matrix2 = np.reshape(matrix2, (3,1))
    
    coord = transMatrix.T * matrix2
    
    return coord[0], coord[1]

def readImgGetTrans(filename, transMatrix):
    nums = 0
    img = misc.imread(filename)
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]
    
    redChannel = []
    greenChannel = []
    blueChannel = [] 
    for x, y in product(range(218), range(178)):
        x_, y_ = getTransCoord(x, y, transMatrix)
        smallX = int(x_)
        smallY = int(y_)
        largeX = int(x_) + 1
        largeY = int(y_) + 1
        if(x_ < 0 or y_ < 0 or x_ >= 217 or y_ >= 177):
            redChannel.append(0)
            greenChannel.append(0)
            blueChannel.append(0)
            nums += 1
            continue
        redValue = bilinearInterpolation(x_, y_, [tuple([smallX, smallY, red[smallX][smallY]]), \
                                            tuple([smallX, largeY, red[smallX][largeY]]), \
                                            tuple([largeX, smallY, red[largeX][smallY]]), \
                                            tuple([largeX, largeY, red[largeX][largeY]])])
        
        greenValue = bilinearInterpolation(x_, y_, [tuple([smallX, smallY, green[smallX][smallY]]), \
                                            tuple([smallX, largeY, green[smallX][largeY]]), \
                                            tuple([largeX, smallY, green[largeX][smallY]]), \
                                            tuple([largeX, largeY, green[largeX][largeY]])])
        
        
        blueValue = bilinearInterpolation(x_, y_, [tuple([smallX, smallY, blue[smallX][smallY]]), \
                                            tuple([smallX, largeY, blue[smallX][largeY]]), \
                                            tuple([largeX, smallY, blue[largeX][smallY]]), \
                                            tuple([largeX, largeY, blue[largeX][largeY]])])
    
        redChannel.append(redValue)
        greenChannel.append(greenValue)
        blueChannel.append(blueValue)
    
    print(nums)
    return [np.reshape(redChannel, (218, 178)), np.reshape(greenChannel, (218, 178)), np.reshape(blueChannel, (218, 178))]
        

def getFilesForRound(dir, lastEnd):
    ret = []
    before = False
    if(lastEnd == ''):
        before = True
    for i in os.listdir(imgDir):
        if i.endswith(".jpg"):
            if(not before):
                if(i == lastEnd):
                    before = True
                    ret.append(i)
            else:
                ret.append(i)
                if(len(ret) == 22511):
                    return ret, i

def run(file):
    filename = os.path.basename(file)
    print(filename)
    key = int(filename.split('.')[0])
    transImg = readImgGetTrans(i, transMatrix[key])
    return transImg

if __name__ == '__main__':
    transMatrix = dd.io.load('test.h5')
    result = []
    pwd = os.getcwd()
    os.chdir(imgDir)
    lastEnd = ''
    pool = Pool()
    for i in range(9):
        files, lastEnd = getFilesForRound(imgDir, lastEnd)
        affinedImgs = pool.map(run, files)
        result.append(np.mean(affinedImgs, axis = 0))
        
    os.chdir(pwd)
    np.save('meanShape24_24', np.mean(result, axis = 0))
