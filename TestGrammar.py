# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:43:23 2018

@author: 
"""

import numpy as np
import scipy as sp
from GraphGrammarOperation import GraphGrammarOperation
from PartitionData import PartitionData
import matplotlib.pyplot as plt


def printMatrix(mat):
    Mus = mat.diagonal().copy()
    mat[np.diag(Mus > 0)] = 0.2
    plt.imshow(mat)
    np.fill_diagonal(mat, Mus)
    plt.show()


def printMatrices(ElasticMatrices, ElasticMatrix=None):
    if ElasticMatrix is not None:
        print("Start : ")
        printMatrix(ElasticMatrix)
        print("-------------------")
    for i in range(ElasticMatrices.shape[2]):
        printMatrix(ElasticMatrices[:, :, i])


nData = 1000
dim = 2
nNodes = 12
X = sp.rand(nData, dim)
ind = np.random.choice(nData, nNodes)
NodePositions = X[ind, ]
print(NodePositions)
ElasticMatrix = np.array(
    [[0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     [0.10, 0.10, 0.01, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00, 0.10, 0.10, 0.01, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.10, 0.00, 0.00, 0.10, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.10, 0.01, 0.10],
     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00]])
XSquared = np.ndarray.reshape((X**2).sum(axis=1), (nData, 1))
part, dists = PartitionData(X, NodePositions, 100000, XSquared)


GraphGrammarOperation(
        X, NodePositions, ElasticMatrix, part, "bisectedge")
