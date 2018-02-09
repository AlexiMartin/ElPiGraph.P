# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:43:23 2018

@author: Alexis Martin
"""

import numpy as np
from test_code.TestData import graph
from core_algorithm.ElPrincGraph import ElPrincGraph
from computeElasticPrincipalCircle import computeElasticPrincipalCircle as cEPC
from computeElasticPrincipalGraph import computeElasticPrincipalGraph as cEPG
import matplotlib.pyplot as plt
import PCAView as PCAV
# import scipy as sp


def printMatrix(mat, size):
    plt.figure(figsize=(size, size))
    Mus = mat.diagonal().copy()
    mat[np.diag(Mus > 0)] = 0.1
    plt.imshow(mat)
    np.fill_diagonal(mat, Mus)
    plt.show()


def represent2Ddata(filename, dispPrev=False):
    ElasticMatrix = (np.array([[0.0, 0.01, 0],
                               [0.01, 0.1, 0.01],
                               [0, 0.01, 0]]))
    X = np.loadtxt(filename)
    # X = np.vstack((X, sp.rand(100, 2)*2-1))
    NodePositions = X[np.random.choice(X.shape[0], ElasticMatrix.shape[0]), ]
    if dispPrev:
        graph(X, NodePositions, ElasticMatrix.shape[0])
    op = np.array([["bisectedge"],
                   ["bisectedge"]])
    op2 = np.array([["removenode", "shrinkedge"]])
    NodePositions, ElasticMatrix = cEPC(X, 50, newDim=np.array([0, 2]), verbose=False)
    # PCAV.PCAView(NodePositions, 0, X)
    printMatrix(ElasticMatrix, 8)
    return X[:, :2], NodePositions[:, :2], ElasticMatrix.shape[0]
    

graph(*represent2Ddata("test_code/test_data/iris.data"))
