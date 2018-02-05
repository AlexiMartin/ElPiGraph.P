# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:46:32 2018

@author: Alexis Martin
"""

import computeElasticPrincipalGraph as EPG
import numpy as np
import matplotlib.pyplot as plt


def computeElasticPrincipalCircle(data, NumNodes, newDim=None,
                                  drawPCAview=True,
                                  drawAccuracyComplexity=True, drawEnergy=True,
                                  Lambda=0.01, Mu=0.1, InitNodeP=None,
                                  InitEdges=None, growGrammar=None,
                                  shrinkGrammar=None, ComputeMSEP=False,
                                  MaxBlockSize=100000, TrimmingRadius=np.inf,
                                  MaxNumberOfIterations=10, eps=0.01,
                                  verbose=True):
    NodeP = np.zeros((4, data.shape[1]))
    v, u, s = EPG.PCA(data)
    mn = data.mean(axis=0)
    v1 = v[:, 0]/np.linalg.norm(v[:, 0])
    v2 = v[:, 1]/np.linalg.norm(v[:, 1])
    st1 = np.std(u[:, 0], ddof=1)
    st2 = np.std(u[:, 1], ddof=1)
    NodeP[0, :] = mn - np.dot(st1, v1.T) - np.dot(st2, v2.T)
    NodeP[1, :] = mn - np.dot(st1, v1.T) + np.dot(st2, v2.T)
    NodeP[2, :] = mn + np.dot(st1, v1.T) + np.dot(st2, v2.T)
    NodeP[3, :] = mn + np.dot(st1, v1.T) - np.dot(st2, v2.T)
    ed = np.array([[0, 1], [2, 3], [1, 2], [3, 0]])
    return EPG.computeElasticPrincipalGraph(data, NumNodes, newDim,
                                            drawPCAview,
                                            drawAccuracyComplexity,
                                            drawEnergy, Lambda,
                                            Mu, NodeP, ed,
                                            np.array([["bisectedge"]]),
                                            np.array([]), ComputeMSEP,
                                            MaxBlockSize, TrimmingRadius,
                                            MaxNumberOfIterations, eps,
                                            verbose)


X = np.loadtxt("test_code/test_data/iris.data")
plt.scatter(*zip(*X[:, 0:2]))
plt.show()
print(computeElasticPrincipalCircle(X, 50, verbose=False)[1].shape)
