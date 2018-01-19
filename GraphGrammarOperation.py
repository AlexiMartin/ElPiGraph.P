# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:26:31 2018

@author:
"""
import numpy as np


def GraphGrammarOperation(X, NodePositions, ElasticMatrix, partition, Type):
    if Type == "addnode2node":
        return AddNode2Node(X, NodePositions, ElasticMatrix, partition)
    elif Type == "removenode":
        return RemoveNode(NodePositions, ElasticMatrix)
    elif Type == "bisectedge":
        return BisectEdge(NodePositions, ElasticMatrix)
    elif Type == "shrinkedge":
        return ShrinkEdge(NodePositions, ElasticMatrix)
    else:
        raise ValueError("Operation " + Type + " is not defined")


# TODO add pointweights ?
def AddNode2Node(X, NodePositions, ElasticMatrix, partition):
    nNodes = NodePositions.shape[0]
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    indL = Lambda > 0
    Connectivities = indL.sum(axis=0)
    # add pointweigths here if added
    assoc = (np.bincount(partition.ravel()+1, minlength=nNodes+1))[1:]
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack((np.hstack((ElasticMatrix, np.zeros((nNodes, 1)))),
                        np.zeros((1, nNodes+1))))
    niProt = np.arange(nNodes+1)
    niProt[nNodes] = 0
    NodePositionsArray = np.repeat(npProt[:, :, np.newaxis], nNodes, axis=2)
    ElasticMatrices = np.repeat(emProt[:, :, np.newaxis], nNodes, axis=2)
    NodeIndicesArray = np.repeat(niProt[:, np.newaxis], nNodes, axis=1)

    for i in range(nNodes):
        meanL = Lambda[i, indL[i, ]].mean(axis=0)
        ElasticMatrices[nNodes, i, i] = ElasticMatrices[i, nNodes, i] = meanL
        if Connectivities[i] == 1:
            ineighbour = np.nonzero(indL[i, ])[0]
            NewNodePosition = 2*NodePositions[i, ]-NodePositions[ineighbour, ]
            ElasticMatrices[i, i, i] = Mus[ineighbour]

        else:
            if assoc[i] == 0:
                NewNodePosition = NodePositions[indL[:, i]].mean(axis=0)

            else:
                NewNodePosition = X[(partition == i).ravel()].mean(axis=0)

        NodePositionsArray[nNodes, :, i] = NewNodePosition
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray


def RemoveNode(NodePositions, ElasticMatrix):
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    nNodes = ElasticMatrix.shape[0]
    nGraphs = (Connectivities == 1).sum()
    NodePositionsArray = np.zeros((nNodes-1, NodePositions.shape[1], nGraphs))
    ElasticMatrices = np.zeros((nNodes-1, nNodes-1, nGraphs))
    NodeIndicesArray = np.zeros((nNodes-1, nGraphs))
    k = 0
    for i in range(Connectivities.shape[0]):
        if Connectivities[i] == 1:
            newInds = np.concatenate((np.arange(0, i), np.arange(i+1, nNodes)))
            NodePositionsArray[:, :, k] = NodePositions[newInds, :]
            ElasticMatrices[:, :, k] = ElasticMatrix[newInds, newInds]
            NodeIndicesArray[:, k] = newInds
            k += 1
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray


def BisectEdge(NodePositions, ElasticMatrix):
    Mus = ElasticMatrix.diagonal()
    start, stop = np.triu(ElasticMatrix, 1).nonzero()
    nGraphs = start.shape[0]
    nNodes = NodePositions.shape[0]
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack((np.hstack((ElasticMatrix, np.zeros((nNodes, 1)))),
                        np.zeros((1, nNodes+1))))
    niProt = np.arange(nNodes+1)
    niProt[nNodes] = 0
    NodePositionsArray = np.repeat(npProt[:, :, np.newaxis], nGraphs, axis=2)
    ElasticMatrices = np.repeat(emProt[:, :, np.newaxis], nGraphs, axis=2)
    NodeIndicesArray = np.repeat(niProt[:, np.newaxis], nGraphs, axis=1)
    for i in range(nGraphs):
        NewNodePosition = (NodePositions[start[i], ] +
                           NodePositions[stop[i], ]) / 2
        NodePositionsArray[nNodes, :, i] = NewNodePosition
        Lambda = ElasticMatrix[start[i], stop[i]]
        ElasticMatrices[start[i], stop[i], i] = 0
        ElasticMatrices[stop[i], start[i], i] = 0
        ElasticMatrices[start[i], nNodes, i] = Lambda
        ElasticMatrices[nNodes, start[i], i] = Lambda
        ElasticMatrices[nNodes, stop[i], i] = Lambda
        ElasticMatrices[stop[i], nNodes, i] = Lambda
        mu1 = Mus[start[i]]
        mu2 = Mus[stop[i]]
        if mu1 > 0 and mu2 > 0:
            ElasticMatrices[nNodes, nNodes, i] = (mu1 + mu2) / 2
        else:
            ElasticMatrices[nNodes, nNodes, i] = max(mu1, mu2)
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray


def ShrinkEdge(NodePositions, ElasticMatrix):
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    start, stop = np.triu(ElasticMatrix, 1).nonzero()
    nNodes = NodePositions.shape[0]
    ind = np.min(np.hstack((Connectivities[start[np.newaxis].T],
                            Connectivities[stop[np.newaxis].T])), axis=1)
    ind = ind > 1
    start = start[ind]
    stop = stop[ind]
    nGraphs = start.shape[0]
    NodePositionsArray = np.zeros((nNodes-1, NodePositions.shape[1], nGraphs))
    ElasticMatrices = np.zeros((nNodes-1, nNodes-1, nGraphs))
    NodeIndicesArray = np.zeros((nNodes-1, nGraphs))
    for i in range(stop.shape[0]):
        em = ElasticMatrix.copy()
        em[start[i], ] = np.maximum(Lambda[start[i], ], Lambda[stop[i], ])
        em[:, start[i]] = np.maximum(Lambda[:, start[i]], Lambda[:, stop[i]])
        em[start[i], start[i]] = (Mus[start[i]] + Mus[stop[i]]) / 2
        nodep = NodePositions.copy()
        nodep[start[i], :] = (nodep[start[i], :] + nodep[stop[i], :]) / 2
        newInds = np.concatenate((np.arange(0, stop[i]),
                                 np.arange(stop[i]+1, nNodes)))
        NodePositionsArray[:, :, i] = nodep[newInds, ]
        ElasticMatrices[:, :, i] = em.take(newInds, axis=0).take(newInds,
                                                                 axis=1)
        NodeIndicesArray[:, i] = newInds
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray
