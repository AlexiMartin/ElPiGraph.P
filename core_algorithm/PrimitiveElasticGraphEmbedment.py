# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:52:31 2018

@author: Alexis Martin
"""
import numpy as np
import core_algorithm.ComputePrimitiveGraphElasticEnergy as CG
from core_algorithm.PartitionData import PartitionData


# This is the core function for fitting a primitive elastic graph to the data
# Inputs
#   X - is the n-by-m data matrix. Each row corresponds to one data point.
#   NodePositions - is k-by-m matrix of positions of the graph nodes in the
#       same space as X.
#   ElasticMatrix - k-by-k symmetric matrix describing the connectivity and
#       the elastic properties of the graph. Star elasticities (mu
#       coefficients) are presented on the main diagonal (non-zero entries
#       only for star centers), and the edge elasticity moduli are
#       presented out of diagonal.
#   varargin contains Name, Value pairs. Names can be:
#   'MaxNumberOfIterations' with integer number which is maximum number of
#       iterations for EM algorithm.
#   'eps' with real number which is minimal relative change in the node
#       positions to be considered the graph embedded (convergence criteria)
#   'PointWeights' with n-by-1 vector of data point weights
#   'MaxBlockSize' with integer number which is maximum size of the block
#       of the distance matrix when partition the data. This means that
#       maximal size of distance matrix is MaxBlockSize-by-k where k is
#       number of nodes.
#   'verbose' with 1/0 is to display/hide the energy values at each
#       iteration and in the end of the process.
#   'TrimmingRadius' is trimming radius, a parameter required for robust
#       principal graphs
# see: github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs
#   'SquaredX' with n-by-1 vector of squared length of data vectors.
#
# Outputs
#   EmbeddedNodePositions is positions of empbedded nodes
#   ElasticEnergy is total elastic energy
#   partition is n-by-1 vector. partition(i) is number of node which is
#       associated with data point X(i,:).
#   dists is array of squared distances form each data point to nerest
#       node.
#   MSE is mean square error of data approximation.
#   EP is edge potential
#   RP is harmonicity potential
def PrimitiveElasticGraphEmbedment(X, NodePositions, ElasticMatrix,
                                   MaxNumberOfIterations=10, eps=0.01,
                                   PointWeights=None, MaxBlockSize=100000,
                                   verbose=False, TrimmingRadius=np.inf,
                                   SquaredX=None):
    N = X.shape[0]
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    # Auxiliary computations
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)
    if SquaredX is None:
        SquaredX = (X**2).sum(axis=1).reshape((N, 1))
    # Main iterative EM cycle: partition, fit given the partition, repeat
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


# Transforms the ElasticMatrix into the SpringLaplacianMatrix ready
# to be used in the SLAU solving
def ComputeSpringLaplacianMatrix(ElasticMatrix):
    NumberOfNodes = ElasticMatrix.shape[0]
    # first, make the vector of mu coefficients
    Mu = ElasticMatrix.diagonal()
    # create the matrix with edge elasticity moduli at non-diagonal elements
    Lambda = ElasticMatrix - np.diag(Mu)
    # Diagonal matrix of edge elasticities
    LambdaSums = Lambda.sum(axis=0)
    # E matrix (contribution from edges) is simply weighted Laplacian
    E = np.diag(LambdaSums) - Lambda
    # matrix S (contribution from stars) is composed of Laplacian for
    # positive strings (star edges) with elasticities mu/k, where k is the
    # order of the star, and Laplacian for negative strings with
    # elasticities -mu/k^2. Negative springs connect all star leafs in a
    # clique.
    StarCenterIndices = np.nonzero(Mu > 0)[0]
    S = np.zeros((NumberOfNodes, NumberOfNodes))
    for i in range(np.size(StarCenterIndices)):
        Spart = np.zeros((NumberOfNodes, NumberOfNodes))
        # leaves indices
        leaves = Lambda.take(StarCenterIndices[i], axis=1) > 0
        # order of the star
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


# ComputeWeightedAverage calculate NodeClusterCentres as weighted averages
# of points from matrix X.
#
# Inputs
#    X is n-by-m matrix of data points where each row corresponds to one
#        observation.
#    partition is n-by-1 (column) vector of node numbers. This vector
#        associate data points with Nodes.
#    PointWeights is n-by-1 (column) vector of point weights.
#    NumberOfNodes is number of nodes to calculate means.
#
# Important! if there is no point associated with node then coordinates of
# this node centroid are zero.
def ComputeWeightedAverage(X, partition, PointWeights, NumberOfNodes):
    X = X * PointWeights
    # Auxiliary calculations
    M = X.shape[1]
    part = partition.ravel() + 1
    # Calculate total weights
    TotalWeight = PointWeights.sum()
    # Calculate weights for Relative size
    tmp = np.bincount(part, weights=PointWeights.ravel(),
                      minlength=NumberOfNodes+1)
    NodeClusterRelativeSize = tmp[1:] / TotalWeight
    # To prevent dividing by 0
    tmp[tmp == 0] = 1
    NodeClusterCenters = np.zeros((NumberOfNodes + 1, X.shape[1]))
    for k in range(M):
        NodeClusterCenters[:, k] = np.bincount(part, weights=X[:, k],
                                               minlength=NumberOfNodes+1)/tmp
    return (NodeClusterCenters[1:, ], NodeClusterRelativeSize[np.newaxis].T)


# Solves the SLAU to find new node positions
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
