# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:06:02 2018

@author: Alexis Martin
"""

import numpy as np


def computeRobustElasticPrincipalGraph(data, NumNodes,  TrimmingRadius,
                                       newDim=None, drawPCAview=True,
                                       drawAccuracyComplexity=True,
                                       drawEnergy=True, Lambda=0.01, Mu=0.1,
                                       InitNodeP=None, InitEdges=None,
                                       growGrammar=None, shrinkGrammar=None,
                                       ComputeMSEP=False, MaxBlockSize=100000,
                                       MaxNumberOfIterations=10, eps=0.01,
                                       verbose=True):
    nodeP = np.zeros((2, data.shape[1]))
    ed = np.array([[0, 1]])
    NumberOfSamples = np.floor(data.shape[0]/10)
    if NumberOfSamples > 1000:
        NumberOfSamples = 1000
    sampling = np.random.permutation(range(data.shape[0]))[:NumberOfSamples]
    

