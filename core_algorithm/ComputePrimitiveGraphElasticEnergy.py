# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:00:53 2018

@author:
"""

import numpy as np


def ComputePrimitiveGraphElasticEnergy(NodePositions, ElasticMatrix, dists):
    MSE = dists.sum() / np.size(dists)
    Mu = ElasticMatrix.diagonal()
    Lambda = np.triu(ElasticMatrix, 1)
    StarCenterIndices = np.nonzero(Mu > 0)[0]
    (row, col) = Lambda.nonzero()
    dev = NodePositions.take(row, axis=0) - NodePositions.take(col, axis=0)
    L = Lambda[Lambda > 0]
    EP = sum(L.flatten('F')*np.sum(dev**2, axis=1))
    indL = Lambda+Lambda.transpose() > 0
    RP = 0
    for i in range(np.size(StarCenterIndices)):
        leaves = indL.take(StarCenterIndices[i], axis=1)
        K = sum(leaves)
        dev = (NodePositions.take(StarCenterIndices[i], axis=0) -
               sum(NodePositions[leaves])/K)
        RP = RP + Mu[StarCenterIndices[i]] * sum(dev ** 2)
    ElasticEnergy = MSE + EP + RP
    return ElasticEnergy, MSE, EP, RP
