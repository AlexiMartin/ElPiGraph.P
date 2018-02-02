# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:43:23 2018

@author: Alexis Martin
"""

import numpy as np
from test_code.TestData import graph
from core_algorithm.ElPrincGraph import ElPrincGraph
import matplotlib.pyplot as plt
# import scipy as sp


def printMatrix(mat, size):
    plt.figure(figsize=(size, size))
    Mus = mat.diagonal().copy()
    mat[np.diag(Mus > 0)] = 0.1
    plt.imshow(mat)
    np.fill_diagonal(mat, Mus)
    plt.show()


def printMatrices(ElasticMatrices, ElasticMatrix=None, size=8):
    if ElasticMatrix is not None:
        print("Start : ")
        printMatrix(ElasticMatrix, size)
        print("-------------------")
    for i in range(ElasticMatrices.shape[2]):
        printMatrix(ElasticMatrices[:, :, i], size)


def represent2Ddata(filename, dispPrev=False):
    ElasticMatrix = (np.array([[0.0, 0.01, 0],
                               [0.01, 0.1, 0.01],
                               [0, 0.01, 0]]))
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
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.10, 0.0]]))
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
    op = np.array([["bisectedge"],
                   ["bisectedge"]])
    op2 = np.array([["removenode", "shrinkedge"]])
    """
    for i in range(15):
        print("Computing operation ", i+1)
        NodePositions, ElasticMatrix, partition, dists = (
            ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                              op))
        NodePositions, ElasticMatrix, partition, dists = (
            ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                              op2))
        NodePositions, ElasticMatrix, partition, dists = (
            ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                              op))
        graph(X, NodePositions, ElasticMatrix.shape[0])
        plt.show()
    """
    NodePositions, ElasticMatrix = ElPrincGraph(X, 50, 0.01, 0.1, verbose=False)

    printMatrix(ElasticMatrix, 8)
    return X, NodePositions, ElasticMatrix.shape[0]
    

graph(*represent2Ddata("test_code/test_data/cells.data"))
