# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:02:30 2018

@author: Alexis Martin
"""
from PartitionData import PartitionData
from GraphGrammarOperation import GraphGrammarOperation
from PrimitiveElasticGraphEmbedment import PrimitiveElasticGraphEmbedment
import numpy as np


# TODO add pointweights ?
def ApplyOptimalGraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                      opTypes, MaxBlockSize=100000,
                                      verbose=False, TrimmingRadius=np.inf,
                                      MaxNumberOfIterations=10, eps=0.01):
    TrimmingRadius = TrimmingRadius**2
    SquaredX = (X**2).sum(axis=1).reshape((X.shape[0], 1))
    partition, _ = PartitionData(X, NodePositions, MaxBlockSize, SquaredX,
                                 TrimmingRadius)
    NodePositionsArrayAll, ElasticMatricesAll, NodeIndicesArrayAll = (
                GraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                      partition, opTypes[0]))
    for i in range(len(opTypes)-1):
        NodePositionsArray, ElasticMatrices, NodeIndicesArray = (
                GraphGrammarOperation(X, NodePositions, ElasticMatrix,
                                      partition, opTypes[i+1]))
        NodePositionsArrayAll = np.concatenate((NodePositionsArrayAll,
                                               NodePositionsArray), axis=2)
        ElasticMatricesAll = np.concatenate((ElasticMatricesAll,
                                             ElasticMatrices), axis=2)
        NodeIndicesArrayAll = np.concatenate((NodeIndicesArrayAll,
                                              NodeIndicesArray), axis=1)
    minEnergy = np.inf
    for i in range(NodeIndicesArrayAll.shape[1]):
        EM = ElasticMatricesAll[:, :, i]
        Mus = EM.diagonal().copy()
        EM1 = EM - np.diag(Mus)
        inds = (EM1 > 0).sum(axis=0) == 1
        Mus[inds] = 0
        EM = EM1 + np.diag(Mus)
        # TODO add pointweights ?
        nodep, ElasticEnergy, part, dist, *_ = (
                PrimitiveElasticGraphEmbedment(X,
                                               NodePositionsArrayAll[:, :, i],
                                               EM, MaxNumberOfIterations, eps,
                                               None, MaxBlockSize, verbose,
                                               TrimmingRadius, SquaredX))
        if(ElasticEnergy < minEnergy):
            minEnergy = ElasticEnergy
            NewNodePositions = nodep
            partition = part
            dists = dist
            NewElasticMatrix = EM
    return NewNodePositions, NewElasticMatrix, partition, dists
