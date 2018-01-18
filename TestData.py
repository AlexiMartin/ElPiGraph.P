# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:45:02 2018

@author:
"""
from PrimitiveElasticGraphEmbedment import PrimitiveElasticGraphEmbedment
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def graph(X, NodePositions, nNodes, partition=None, col=None, size=5):
    A = [None] * nNodes
    if col is None:
        col = ['c', 'r', 'b', 'g', 'y', 'm', 'k']
    plt.figure(figsize=(size, size))
    if partition is None:
        plt.plot(*zip(*X), marker='.', ls='', color=col[0])
    else:
        for i in range(nNodes):
            A[i] = X[partition[:, 0] == i]
            plt.plot(*zip(*A[i]), marker='.', ls='',
                     color=col[i % np.size(col)])
    plt.plot(*zip(*NodePositions), marker='o', color='r', ls='')


def uniformRandom():
    nData = 1000
    dim = 2
    nNodes = 12
    X = sp.rand(nData, dim)
    ind = np.random.choice(nData, nNodes)
    NodePositions = X[ind, ]
    ElasticMatrix = np.array([[0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0.1, 0.1, 0.01, 0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0.1, 0.1, 0.01, 0.1, 0.1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0.1, 0, 0, 0.1, 0, 0.01, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.01, 0.1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.01]])
    EmbeddedNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP = (
        PrimitiveElasticGraphEmbedment(X, NodePositions, ElasticMatrix))
    return X, EmbeddedNodePositions, nNodes, partition


def line2Data(filename, dispPrev=False):
    ElasticMatrix = np.array([[0.0, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    X = np.array([[0, 1]])
    with open(filename) as F:
        for line in F:
            X = np.vstack((X, (np.array(line.split('\t')))[:2]))
    X = X[1:].astype(float)
    NodePositions = X[np.random.choice(X.shape[0], ElasticMatrix.shape[0]), ]
    if dispPrev:
        graph(X, NodePositions, ElasticMatrix.shape[0])
    EmbeddedNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP = (
            PrimitiveElasticGraphEmbedment(X, NodePositions, ElasticMatrix,
                                           verbose=True))
    return X, EmbeddedNodePositions, ElasticMatrix.shape[0], partition


# graph(*line2Data("tree23.data", True), col=['b'])
