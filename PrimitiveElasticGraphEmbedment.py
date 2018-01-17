# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:52:31 2018

@author:
"""
import numpy as np
import scipy as sp
import math
import ComputePrimitiveGraphElasticEnergy as CG
from PartitionData import PartitionData


def PrimitiveElasticGraphEmbedment(X, NodePositions, ElasticMatrix,
                                   MaxNumberOfIterations=10, eps=0.01,
                                   PointWeights=None, MaxBlockSize=100000,
                                   verbose=False, TrimmingRadius=math.inf,
                                   SquaredX=None):
    N = X.shape[0]
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    partition = []
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)
    if SquaredX is None:
        SquaredX = (X**2).sum(axis=1).reshape((N, 1))
    for i in range(MaxNumberOfIterations):
        if verbose:
            partition, dists = PartitionData(X, NodePositions, MaxBlockSize,
                                             SquaredX, TrimmingRadius)
            ElasticEnergy, MSE, EP, RP = CG.ComputePrimitiveGraphElasticEnergy(
                    NodePositions, ElasticMatrix, dists)
        else:
            partition, dists = PartitionData(X, NodePositions, MaxBlockSize,
                                             SquaredX, TrimmingRadius)
        NewNodePositions = FitGraph2DataGivenPartition(
                X, PointWeights, SpringLaplacianMatrix, partition)
        diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions)
        if verbose:
            print("Iteration ", (i+1), " difference of node position=", diff,
                  ", Energy=", ElasticEnergy, ", MSE=", MSE, ", EP=", EP,
                  ", RP=", RP)
        if diff < eps:
            break
        NodePositions = NewNodePositions
    partition, dists = PartitionData(X, NodePositions, MaxBlockSize,
                                     SquaredX, TrimmingRadius)
    ElasticEnergy, MSE, EP, RP = CG.ComputePrimitiveGraphElasticEnergy(
                    NodePositions, ElasticMatrix, dists)
    if verbose:
        print("E=", ElasticEnergy, ", MSE=", MSE, ", EP=", EP, ", RP=", RP)
    EmbeddedNodePositions = NodePositions
    return (EmbeddedNodePositions, ElasticEnergy, partition, dists,
            MSE, EP, RP)


def ComputeSpringLaplacianMatrix(ElasticMatrix):
    E = sp.sparse.csgraph.laplacian(ElasticMatrix)
    NumberOfNodes = np.size(ElasticMatrix, axis=0)
    Mu = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    StarCenterIndices = np.nonzero(Mu > 0)[0]
    S = np.zeros((NumberOfNodes, NumberOfNodes))
    for i in range(np.size(StarCenterIndices)):
        Spart = np.zeros((NumberOfNodes, NumberOfNodes))
        leaves = Lambda.take(StarCenterIndices[i], axis=1) > 0
        K = leaves.sum()
        Spart[StarCenterIndices[i], StarCenterIndices[i]] = (
                Mu[StarCenterIndices[i]])
        Spart[StarCenterIndices[i], leaves] = -Mu[StarCenterIndices[i]]/K
        Spart[leaves, StarCenterIndices[i]] = -Mu[StarCenterIndices[i]]/K
        tmp = np.repeat(leaves[np.newaxis],
                        ElasticMatrix.shape[0], axis=0)
        tmp = np.logical_and(tmp, tmp.transpose())
        Spart[tmp] = Mu[StarCenterIndices[i]]/(K**2)
        S = S + Spart
    return E + S


def ComputeWeightedAverage(X, partition, PointWeights, NumberOfNodes):
    X = X * PointWeights
    M = X.shape[1]
    TotalWeight = PointWeights.sum()
    part = partition.ravel() + 1
    tmp = np.bincount(part, weights=PointWeights.ravel(),
                      minlength=NumberOfNodes+1)
    NodeClusterRelativeSize = tmp[1:] / TotalWeight
    tmp[tmp == 0] = 1
    NodeClusterCenters = np.zeros((NumberOfNodes + 1, X.shape[1]))
    for k in range(M):
        NodeClusterCenters[:, k] = np.bincount(part, weights=X[:, k],
                                               minlength=NumberOfNodes+1)/tmp
    return (NodeClusterCenters[1:, ], NodeClusterRelativeSize[np.newaxis].T)


def FitGraph2DataGivenPartition(X, PointWeights, SpringLaplacianMatrix,
                                partition):
    NumberOfNodes = np.size(SpringLaplacianMatrix, axis=0)
    NodeClusterCenters, NodeClusterRelativeSize = (
            ComputeWeightedAverage(X, partition, PointWeights, NumberOfNodes))
    SLAUMatrix = np.diag(NodeClusterRelativeSize.T[0]) + SpringLaplacianMatrix
    NewNodePositions = np.linalg.solve(SLAUMatrix, NodeClusterRelativeSize *
                                       NodeClusterCenters)
    return NewNodePositions


def ComputeRelativeChangeOfNodePositions(NodePositions, NewNodePositions):
    return max(np.sum((NodePositions - NewNodePositions)**2, axis=1) /
               np.sum(NewNodePositions**2, axis=1))
