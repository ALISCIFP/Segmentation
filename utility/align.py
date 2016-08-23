# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 14:27:46 2016

@author: Lengyue Chen
"""

import os  
import re
import numpy as np
import h5py
import deepdish as dd
from sympy import *

fileName = "meanShape.txt"
datafileName = "list_landmarks_celeba.txt"

def Affine_Fit( from_pts, to_pts ):
    """Fit an affine transformation to given point sets.
      More precisely: solve (least squares fit) matrix 'A'and 't' from
      'p ~= A*q+t', given vectors 'p' and 'q'.
      Works with arbitrary dimensional vectors (2d, 3d, 4d...).

      Written by Jarno Elonen <elonen@iki.fi> in 2007.
      Placed in Public Domain.

      Based on paper "Fitting affine and orthogonal transformations
      between two sets of points, by Helmuth Späth (2003)."""

    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q)<1:
        print("from_pts and to_pts must be of same size.")
        return false

    dim = len(q[0]) # num of dimensions
    if len(q) < dim:
        print("Too few points => under-determined system.")
        return false

    # Make an empty (dim) x (dim+1) matrix and fill it
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]
    for j in range(dim):
        for k in range(dim+1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # Make an empty (dim+1) x (dim+1) matrix and fill it
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim+1):
            for j in range(dim+1):
                Q[i][j] += qt[i] * qt[j]

    # Ultra simple linear system solver. Replace this if you need speed.
    def gauss_jordan(m, eps = 1.0/(10**10)):
      """Puts given matrix (2D array) into the Reduced Row Echelon Form.
         Returns True if successful, False if 'm' is singular.
         NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
         Written by Jarno Elonen in April 2005, released into Public Domain"""
      (h, w) = (len(m), len(m[0]))
      for y in range(0,h):
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
          if abs(m[y2][y]) > abs(m[maxrow][y]):
            maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        if abs(m[y][y]) <= eps:     # Singular?
          return False
        for y2 in range(y+1, h):    # Eliminate column y
          c = m[y2][y] / m[y][y]
          for x in range(y, w):
            m[y2][x] -= m[y][x] * c
      for y in range(h-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]
        for y2 in range(0,y):
          for x in range(w-1, y-1, -1):
            m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):       # Normalize row y
          m[y][x] /= c
      return True

    # Augement Q with c and solve Q * a' = c by Gauss-Jordan
    M = [ Q[i] + c[i] for i in range(dim+1)]
    if not gauss_jordan(M):
        print("Error: singular matrix. Points are probably coplanar.")
        return false

    # Make a result object
    class Transformation:
        """Result object that represents the transformation
           from affine fitter."""

        def To_Str(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    M[i][j+dim+1]
                    str +="x%d * %f + " % (i, M[i][j+dim+1])
                str += "%f" % M[dim][j+dim+1]
                res += str + "\n"
            return res

        def Transfor_Matrix(self):
            tmp = []
            ret = []
            for j in range(dim):
                tmp.append([])
                for i in range(dim):
                    tmp[j].append(M[i][j+dim+1])
                tmp[j].append(M[dim][j+dim+1])
            return np.matrix(tmp).transpose()
            #return tmp


        def Transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j+dim+1]
                res[j] += M[dim][j+dim+1]
            return res
    return Transformation()


def readMeanShape(file_name):
	mean_shape = np.loadtxt(file_name)
	return mean_shape

def readAnnotationFile(fileName):
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
                  
            leftEye = np.array([leftEyeX, leftEyeY])
            rightEye = np.array([rightEyeX, rightEyeY])
            nose = np.array([noseX, noseY])
            leftMouth = np.array([leftMouthX, leftMouthY])
            rightMouth = np.array([rightMouthX, rightMouthY])
            
            allLandmarksPerPicture = [leftEye, rightEye, nose, leftMouth, rightMouth]
            #diagonlaize
            allLandmarks.append(allLandmarksPerPicture)
            
        allLandmarks = np.array(allLandmarks)
        return allLandmarks

def convert(list_of_list):
    landmarks_tuple = []
    for point in list_of_list:
        tmp = tuple(point)
        landmarks_tuple.append(tmp)
    landmarks_tuple = tuple(landmarks_tuple)

    return landmarks_tuple


if __name__ == "__main__":
    mean_shape = readMeanShape(fileName)
    allLandmarks = readAnnotationFile(datafileName)
    mean_shape_tuple = convert(mean_shape)
    X_i_dictionary = {}
    for idx,landmark in enumerate(allLandmarks):
        #construct tuple
        landmark_tuple =  convert(landmark)
        trn = Affine_Fit(landmark_tuple,mean_shape_tuple)    
        X_i_dictionary[idx] = trn.Transfor_Matrix()
       # print(trn.Transfor_Matrix())
	print (trn.To_Str())
    #不是很确定这个save的h5对不对
    dd.io.save('test.h5', X_i_dictionary, compression=None) 


    
        
    
    
    
    
    

    
    
 
