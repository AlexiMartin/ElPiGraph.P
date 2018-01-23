# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:43:23 2018

@author: Alexis Martin
"""

import numpy as np
import scipy as sp
from TestData import graph
from ApplyOptimalGraphGrammarOperation import ApplyOptimalGraphGrammarOperation
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


def line2Data(filename, dispPrev=False):
    ElasticMatrix = np.array([[0, 0.01, 0],
                              [0.01, 0.05, 0.01],
                              [0, 0.01, 0]])
    """
    np.array(
            [[0.0, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.1, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0.1, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0.1, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0.1, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0.1, 0.01, 0.1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0.1, 0.01, 0.1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0.1, 0.01, 0.1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0.1, 0.01, 0.1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.01, 0.1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.01, 0.1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.01, 0.1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.01, 0.1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.10, 0.0]])
    """
    X = np.array([[0, 1]])
    with open(filename) as F:
        for line in F:
            X = np.vstack((X, (np.array(line.split('\t')))[:2]))
    X = X[1:].astype(float)
    # X = np.vstack((X, sp.rand(100, 2)*2-1))
    NodePositions = X[np.random.choice(X.shape[0], ElasticMatrix.shape[0]), ]
    if dispPrev:
        graph(X, NodePositions, ElasticMatrix.shape[0])
    op = np.array([["addnode2node", "bisectedge"]])
    op2 = np.array(["removenode", "shrinkedge"])
    for i in range(27):
        print("Computing operation ", i+1)
        NodePositions, ElasticMatrix, partition, dists = (
            ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                              op[0]))
        NodePositions, ElasticMatrix, partition, dists = (
            ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                              op2))
        NodePositions, ElasticMatrix, partition, dists = (
            ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                              op[0]))
        graph(X, NodePositions, ElasticMatrix.shape[0])
        plt.show()
    printMatrix(ElasticMatrix)
    return X, NodePositions, ElasticMatrix.shape[0]


"""
nData = 1000
dim = 2
nNodes = 12
X = sp.rand(nData, dim)
ind = np.random.choice(nData, nNodes)
NodePositions = X[ind, ]
# print(NodePositions)
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
op = ["addnode2node", "bisectedge"]
op2 = ["removenode", "shrinkedge"]
print(NodePositions)
"""
graph(*line2Data("tree23.data"))
