# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:34:45 2016

@author: Kangyan Zhou
"""

from __future__ import print_function

import os  
import re
import numpy as np
from PIL import Image
import h5py

fold1Path = "./medical_data/fold1/"
fold2Path = "./medical_data/fold2/"
patternBracket1 = '\[(.+)\]'
patternBracket2 = '\{(.+)\}'

labelDef = {"background" : 0, "left clavicle": 1, "left lung": 2, "heart": 3, "right clavicle": 4, "right lung": 5}

def loadData_Landmarks(isTraining):
    global labelDef
    if(isTraining):
        currPath = fold1Path + 'landmarks/'
    else:
        currPath = fold2Path + 'landmarks/'
    
    labels = {}
    labelPerPicture = []
    filenames = []
    for filename in os.listdir(currPath):
        with open(currPath + filename, 'r') as file:
            label = 0
            for line in file:
                #getting label info
                m = re.search(patternBracket1, line)
                if(m):
                    info = m.group(1).lower().split("=")[1]
                    if(info not in labelDef.keys()):
                        continue
                    
                    label = labelDef[info]
                
                # get point info
                m = re.search(patternBracket2, line)
                if(m):
                    info = m.group(1)
                    pair = info.split(',')
                    x = float(pair[0]) - 1
                    y = float(pair[1]) - 1
                    
                    temp = []
                    temp.append(x)
                    temp.append(y)
                    temp.append(label)
                    labelPerPicture.append(temp)
        
        filenames.append(filename.split(".")[0] + '.jpg')
    
    if(isTraining):
        os.chdir("./Medical Image Labels/train")
        name = 'train'
    else:
        os.chdir("./Medical Image Labels/test")
        name = 'test'
    
    print(len(filenames))
    print(np.shape(np.array(labelPerPicture)))
    with h5py.File(name + '.h5', 'w') as file:
        file['data'] = np.array(filenames, dtype = np.string_)
        if(name == 'train'):
            file['label'] = np.array(labelPerPicture, dtype = np.float32).reshape((124, 166, 3))
        else:
            file['label'] = np.array(labelPerPicture, dtype = np.float32).reshape((123, 166, 3))
    
    with open(name + '_data.txt', 'w') as file:
        file.write('src/caffe/test/test_data/test.h5\n')
    
    with h5py.File(name + '.h5', 'r') as file:
        test1 = file['data'].value
        test2 = file['label'].value
        
    print(test1[0])
    print(test2[0])
    
    return labels

def loadData_Points(isTraining):
    if(isTraining):
        currPath = fold1Path + 'points/'
    else:
        currPath = fold2Path + 'points/'
        
    allFileFeatures = {}
    for fn in os.listdir(currPath):
        oneFileInfo = []
        newFeature = []
        points = []
        with open(currPath + fn, 'r') as file:
            for line in file:
                if(line.startswith(";")):
                    continue
                
                if(line.startswith("{")):
                    points = []
                    newFeature = []
                    
                if(line.startswith("}")):
                    newFeature.append(points)
                    oneFileInfo.append(newFeature)
                    
                #getting label info
                m = re.search(patternBracket1, line)
                if(m):
                    info = m.group(1)
                    newFeature.append(info.split('=')[1])
                
                # get point info
                m = re.search(patternBracket2, line)
                if(m):
                    info = m.group(1)
                    pair = info.split(',')
                    pair[0] = float(pair[0])
                    pair[1] = float(pair[1])
                    points.append(pair)
                
        allFileFeatures[fn] = oneFileInfo
    
    return allFileFeatures

def loadData_Images(isTraining):
    if(isTraining):
        currPath = fold1Path + 'masks/'
    else:
        currPath = fold2Path + 'masks/'
    
    images = {}
    for dirName in os.listdir(currPath):
        currCategoryPath = currPath + dirName + '/'
        for name in os.listdir(currCategoryPath):
            img = Image.open(currCategoryPath + name)
            arr = np.array(img)
            if(name not in images.keys()):
                images[name] = arr
                continue
            images[name] += arr
    
    os.chdir("./Medical Pictures")
    for filename, image in images.items():
        image[image > 255] = 255
        output = Image.fromarray(image)
        output.save(filename.split(".")[0] + ".jpeg")
        
    return images

temp1 = loadData_Landmarks(True)