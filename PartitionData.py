# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4  2018

@author : Alexis Martin
"""

import numpy as np


# Partition the data by proximity to graph nodes
# (same step as in K-means EM procedure)
#
# Inputs:
#   X is n-by-m matrix of datapoints with one data point per row. n is
#       number of data points and m is dimension of data space.
#   NodePositions is k-by-m matrix of embedded coordinates of graph nodes,
#       where k is number of nodes and m is dimension of data space.
#   MaxBlockSize integer number which defines maximal number of
#       simultaneously calculated distances. Maximal size of created matrix
#       is MaxBlockSize-by-k, where k is number of nodes.
#   SquaredX is n-by-1 vector of data vectors length: SquaredX = sum(X.^2,2);
#   TrimmingRadius (optional) is squared trimming radius.
#
# Outputs
#   partition is n-by-1 vector. partition[i] is number of the node which is
#       associated with data point X[i, ].
#   dists is n-by-1 vector. dists[i] is squared distance between the node with
#       number partition[i] and data point X[i, ].
def PartitionData(X, NodePositions, MaxBlockSize, SquaredX,
                  TrimmingRadius=np.inf):
    n = X.shape[0]
    partition = np.zeros((n, 1), dtype=int)
    dists = np.zeros((n, 1))
    # Calculate squared length of centroids
    cent = NodePositions.T
    centrLength = (cent**2).sum(axis=0)
    # Process partitioning without trimming
    for i in range(0, n, MaxBlockSize):
        # Define last element for calculation
        last = i+MaxBlockSize
        if last > n:
            last = n
        # Calculate distances
        d = SquaredX[i:last] + centrLength-2*np.dot(X[i:last, ], cent)
        tmp = d.argmin(axis=1)
        partition[i:last] = tmp[:, np.newaxis]
        dists[i:last] = d[np.arange(d.shape[0]), tmp][:, np.newaxis]
    # Apply trimming
    if TrimmingRadius is not np.inf:
        ind = dists > TrimmingRadius
        partition[ind] = 0
        dists[ind] = TrimmingRadius
    return partition, dists
