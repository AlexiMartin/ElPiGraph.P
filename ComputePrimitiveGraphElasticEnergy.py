# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:00:53 2018

@author:
"""

import numpy


def ComputePrimitiveGraphElasticEnergy(NodePositions, ElasticMatrix, dists):
    MSE = dists.sum() / numpy.size(dists)
    Mu = ElasticMatrix.diagonal()
    Lambda = numpy.triu(ElasticMatrix, 1)
    StarCenterIndices = numpy.nonzero(Mu > 0)[0]  # indices
    (row, col) = Lambda.nonzero()
    dev = NodePositions.take(row, axis=0) - NodePositions.take(col, axis=0)
    L = Lambda[Lambda > 0]
    EP = sum(L.flatten('F')*numpy.sum(dev**2, axis=1))
    indL = Lambda+Lambda.transpose() > 0
    RP = 0
    for i in range(numpy.size(StarCenterIndices)):
        leaves = indL.take(StarCenterIndices[i], axis=1)
        K = sum(leaves)
        dev = (NodePositions.take(StarCenterIndices[i], axis=0) -
               sum(NodePositions[leaves])/K)
        RP = RP + Mu[StarCenterIndices[i]] * sum(dev ** 2)
    ElasticEnergy = MSE + EP + RP
    return ElasticEnergy, MSE, EP, RP
