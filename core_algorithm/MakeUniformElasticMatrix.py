# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:23:57 2018

@author: Alexis Martin
"""
import numpy as np


def MakeUniformElasticMatrix(Edges, Lambda, Mu):
    NumberOfNodes = Edges.max()+1
    ElasticMatrix = np.zeros((NumberOfNodes, NumberOfNodes))
    for i in range(Edges.shape[0]):
        ElasticMatrix[Edges[i][0], Edges[i][1]] = Lambda
        ElasticMatrix[Edges[i][1], Edges[i][0]] = Lambda
    Connect = (ElasticMatrix > 0).sum(axis=0)
    ind = Connect > 1
    Mus = np.zeros((NumberOfNodes, 1))
    Mus[ind] = Mu
    ElasticMatrix = ElasticMatrix + np.diag(Mus.ravel())
    return ElasticMatrix
