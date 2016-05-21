# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.spatial import procrustes
from numpy.core.umath_tests import matrix_multiply
from numpy.linalg import svd, eig
from sklearn.decomposition import PCA
import random
import h5py
import sys
import os
print(os.getcwd())
sys.path.insert(0, '../data/')

from data_loading import loadTrainingLandmarks, loadTestingLandmarks

a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
c = np.array([[5, -3], [2, -6], [3, -5], [8, -6]], 'd')
f = np.array([a, b, c])

def procrustes2d_IncorrectInput(origin):
    f = np.copy(origin)
    print(np.shape(f))
    numberOfShapes = np.shape(f)[0]
    diff = np.inf
    oldDiff = 0
    newDiff = np.inf
    timesToConverge = 0
#    while(abs(oldDiff - newDiff) > 0.01):
    while(abs(diff) > 0.01):
        timesToConverge += 1
    
        # zero mean normalize input aray
        mean = np.mean(f, axis = 1, keepdims=True)
#        print(mean)
#        print(np.shape(mean))
        #f = (f.T - mean).T
        f -= mean
#        print("f:", f)
    
        #pick radom element as pivot
        #scale pivot element to 1
        index = random.randint(0, numberOfShapes - 1)
        pivot = f[index]
#        print(pivot)
#        print(np.shape(pivot))
        scale = np.linalg.norm(pivot)
#        print("scale", scale)
        pivot /= scale
#        print(pivot)
        f[index] = pivot
#        scale1 = np.sqrt(np.sum(np.power(pivot, 2))/np.shape(pivot)[1])
#        print(scale1)
    
        # do the rotation
        bot = np.einsum('kij, ij...->ki', f, pivot)
#        print(bot)
        bot1 = np.sum(bot, axis = 1)
#        print(bot1)
        pivot = np.roll(pivot, 1, axis = 0)
        pivot[1] = -pivot[1]
#        print(pivot)
        top = np.einsum('kij, ij...->ki', f, pivot)
#        print(top)
        top1 = np.sum(top, axis = 1)
#        print(top1)
        theta = np.arctan2(top1, bot1)
#        print(theta)

        sine = np.sin(theta)
#        print('sine:', sine)
        cosine = np.cos(theta)
#        print('cosine:', cosine)
        transformMatrix = np.zeros((np.shape(f)[0] , 4))

        #build transfor matrix
        transformMatrix[:, 0] = cosine
        transformMatrix[:, 1] = -sine
        transformMatrix[:, 2] = sine
        transformMatrix[:, 3] = cosine
#        print("before reshape", transformMatrix)
        transformMatrix = transformMatrix.reshape(np.shape(f)[0], 2, 2)
#        print("after reshape", transformMatrix)

        update = matrix_multiply(transformMatrix, f)
#        print(update[index] == pivot)
#        print("update:", update)
        
        f = update
        newMean = np.sum(f, axis = 0, keepdims=True)/numberOfShapes
#        print(newMean)
        diff = np.sqrt(np.sum(np.square(f-newMean)))
#        print("diff", diff)
#        print(timesToConverge)
        if(timesToConverge > 10000):
            break
        oldDiff = newDiff
        newDiff = diff
    
    print(timesToConverge)
    print(diff)
    print(f)
    return f
    
def procrustes2d_CorrectInput(origin):
    f = np.copy(origin)
    numberOfShapes = np.shape(f)[0]
    
    diff = np.inf
    timesToConverge = 0        
    while(timesToConverge < 1000):
#    while(abs(diff) > 0.01):
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
#        print(diff)
        timesToConverge += 1
    
    print(diff)
    return f
    
#print(procrustes2d_CorrectInput(f))

temp = loadTrainingLandmarks()
for key, value in temp.items():
    print(key)
    matrixAfterPro = procrustes2d_CorrectInput(np.array(value))
    temp[key] = matrixAfterPro

#f = np.array([[[0, 1, 2], [10, 11, 12], [0, 1, 2]], [[3, 4, 5], [6, 7, 8], [0, 1, 2]]], dtype = np.float)
def procrustes3d(f):
    numberOfShapes = np.shape(f)[0]    
    timesToConverge = 0
    diff = np.inf    
    while(diff > 1):
        prevf = np.copy(f)
        
        # zero mean normalize input aray
        mean = np.mean(np.mean(f, axis = 0), axis = 1).reshape(3, 1)
#        print(mean)
#        print(np.shape(mean))
#        f = (f.T - mean).T
        f = f - mean

        #pick radom element as pivot
        #scale pivot element to 1
        index = random.randint(0, numberOfShapes - 1)
        pivot = f[index]
        #print(pivot)
        #print(np.shape(pivot))
        scale = np.sqrt(np.sum(np.power(pivot-mean, 2))/np.shape(pivot)[1])
        #print(scale)
        pivot = (pivot-mean)/scale
        print(pivot)
        #scale1 = np.sqrt(np.sum(np.power(pivot, 2))/np.shape(pivot)[1])
        #print(scale1)    
        
        timesToConverge += 1  
        f[index] = pivot
        # do the rotation
        for i in range(0, np.shape(f)[0]):
            if(i == index):
                continue
#            print(f[i])
            u, s, v = svd(f[i].T * pivot)
            transformMatrix = v * u.T
            f[i] = matrix_multiply(transformMatrix, f[i])
            print(f[i])
        
        diff = np.sqrt(np.sum(np.square(f - prevf)))
        print(diff)
    
    print(timesToConverge)
    return f

#print(procrustes3d(f))

#pca
def pca(matrixAfterPro, metric, isWhiten):
    if(type(metric) != float):
        raise Exception("metric type error!")
        
    shape = np.shape(matrixAfterPro)
    matrixAfterPro = np.reshape(matrixAfterPro, (shape[0], shape[1]*shape[2]))
    
    pca = PCA(n_components = metric, whiten = isWhiten)
    matrixAfterPCA = pca.fit_transform(matrixAfterPro)
    
    return matrixAfterPCA

def getEigenPairs(matrixAfterPro, name, retainPercentage):
    if(type(retainPercentage) != float or type(name) != str):
        raise Exception("type error!")
    
    shape = np.shape(matrixAfterPro)
    matrixAfterPro = np.reshape(matrixAfterPro, (shape[0], shape[1]*shape[2]))
    eigenValues, eigenVectors = eig(matrixAfterPro)
    percentage = eigenValues/np.sum(eigenValues)
    percentageIndexRank = np.argsort(percentage)[::-1]
    
    newEigenVectors = np.array([])
    sortedEigenvalues = []
    currPercentage = 0
    for i, index in enumerate(percentageIndexRank):
        newEigenVectors = np.append(newEigenVectors, eigenVectors[:, index])
        sortedEigenvalues[i] = eigenValues[index]
        currPercentage += percentage[index]
        if(currPercentage > retainPercentage):
            break
    
    newEigenVectors = newEigenVectors.T
    sortedEigenvalues = np.array(sortedEigenvalues)
    with h5py.File(name + "_" + str(retainPercentage) + '.h5', 'w') as file:
        file['eigenvalue'] = sortedEigenvalues
        file['eigenvector'] = newEigenVectors

shape = np.shape(value)
matrixAfterPro = np.reshape(value, (shape[0], shape[1]*shape[2]))
getEigenPairs(value, 'wtf', 0.98)
print(pca(f, np.float(0.98), False))
