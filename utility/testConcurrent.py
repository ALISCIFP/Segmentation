# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 14:38:46 2016

@author: Kangyan Zhou
"""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import timeit

indexes = [_ for _ in range(100, 260, 1)]

def run(x):
    return [_ for _ in range(x*10)]

def multiprocessVersion():
    with ThreadPoolExecutor(max_workers=12) as executor:
#        final = []
#        returns = []
        
#        executor.map(run, indexes)
        for i in indexes:
            result = executor.submit(run, i).result()
##            print("finish!")
#            returns.append(result)
#            if(len(returns) == 3):
#                final.append(returns)
#                returns = []

def normalVersion():
    for i in indexes:
        run(i)

if __name__ == '__main__':
    print(timeit.timeit(multiprocessVersion, number = 100))
    print(timeit.timeit(normalVersion, number = 100))