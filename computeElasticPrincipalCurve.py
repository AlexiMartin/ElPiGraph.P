# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:53:19 2018

@author: Alexis Martin
"""

import numpy as np
from computeElasticPrincipalGraph import computeElasticPrincipalGraph


def computeElasticPrincipalCurve(data, NumNodes, newDim=None, drawPCAview=True,
                                 drawAccuracyComplexity=True, drawEnergy=True,
                                 Lambda=0.01, Mu=0.1, InitNodeP=None,
                                 InitEdges=None, ComputeMSEP=False,
                                 MaxBlockSize=100000, TrimmingRadius=np.inf,
                                 MaxNumberOfIterations=10, eps=0.01,
                                 verbose=True):
    return computeElasticPrincipalGraph(data, NumNodes, newDim, drawPCAview,
                                        drawAccuracyComplexity, drawEnergy,
                                        Lambda, Mu, InitNodeP, InitEdges,
                                        np.array([["bisectedge"]]),
                                        np.array([]), ComputeMSEP,
                                        MaxBlockSize, TrimmingRadius,
                                        MaxNumberOfIterations, eps, verbose)
