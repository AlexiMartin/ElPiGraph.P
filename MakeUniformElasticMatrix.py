# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:23:57 2018

@author: Alexis Martin
"""
import numpy as np


def MakeUniformElasticMatrix(Edges, Lambda, Mu):
    NumberOfNodes = Edges.max()+1
    ElasticMatrix = np.zeros((NumberOfNodes, NumberOfNodes))
    ind = Edges[:, 0]+1 + (Edges[:, 1]-1)*ElasticMatrix.shape[0]
    ElasticMatrix[np.unravel_index(ind, (ElasticMatrix.shape[0],
                                         ElasticMatrix.shape[0]))] = Lambda
    ElasticMatrix = ElasticMatrix + ElasticMatrix.T
    Connect = (ElasticMatrix > 0).sum(axis=0)
    ind = Connect > 1
    Mus = np.zeros((NumberOfNodes, 1))
    Mus[ind] = Mu
    ElasticMatrix = ElasticMatrix + np.diag(Mus.ravel())
    return ElasticMatrix
