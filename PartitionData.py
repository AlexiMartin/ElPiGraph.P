# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import math


def PartitionData(X, NodePositions, MaxBlockSize, SquaredX,
                  TrimmingRadius=math.inf):
    n = numpy.size(X, 0)
    partition = numpy.zeros((n, 1), dtype=int)
    dists = numpy.zeros((n, 1))
    cent = NodePositions.T
    centrLength = (cent**2).sum(axis=0)
    i = 0
    while i < n:
        last = i+MaxBlockSize
        if last > n:
            last = n
        ind = numpy.arange(i, last)
        d = centrLength-2*numpy.dot(X[ind, ], cent)
        tmp = d.argmin(axis=1)
        partition[ind] = tmp.reshape(last-i, 1)
        dists[ind] = d[numpy.arange(d.shape[0]), tmp].reshape(last-i, 1)
        i += MaxBlockSize
    dists = dists + SquaredX
    ind = dists > TrimmingRadius
    partition[ind] = 0
    dists[ind] = TrimmingRadius
    return partition, dists
