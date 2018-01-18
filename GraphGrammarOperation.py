# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:26:31 2018

@author:
"""
import numpy as np
from PartitionData import PartitionData
import scipy as sp
import matplotlib.pyplot as plt


def GraphGrammarOperation(X, NodePositions, ElasticMatrix, partition, Type):
    if Type == "addnode2node":
        return AddNode2Node(X, NodePositions, ElasticMatrix, partition)
    elif Type == "removenode":
        return RemoveNode(NodePositions, ElasticMatrix)
    elif Type == "bisectedge":
        return Type
    elif Type == "shrinkedge":
        return Type
    else:
        raise ValueError("Operation " + Type + " is not defined")


# TODO add pointweights ?
# OK in theory
def AddNode2Node(X, NodePositions, ElasticMatrix, partition):
    nNodes = NodePositions.shape[0]
    NodePositionsArray = np.zeros((nNodes+1, NodePositions.shape[1], nNodes))
    ElasticMatrices = np.zeros((nNodes+1, nNodes+1, nNodes))
    NodeIndicesArray = np.zeros((nNodes+1, nNodes))
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

    for i in range(nNodes):
        meanL = Lambda[i, indL[i, ]].mean(axis=0)
        NodePositionsArray[:, :, i] = npProt
        ElasticMatrices[:, :, i] = emProt
        NodeIndicesArray[:, i] = niProt
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


def printMatrices(ElasticMatrices, ElasticMatrix=None):
    if ElasticMatrix is not None:
        print("Start : ")
        ElasticMatrix[np.diag(ElasticMatrix.diagonal() > 0)] = 0.2
        plt.imshow(ElasticMatrix)
        plt.show()
    for i in range(ElasticMatrices.shape[2]):
        ElasticMatrices[np.diag(ElasticMatrices[:, :, i].diagonal() > 0),
                        i] = 0.2
        plt.imshow(ElasticMatrices[:, :, i])
        plt.show()


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
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.0]])
XSquared = np.ndarray.reshape((X**2).sum(axis=1), (nData, 1))
part, dists = PartitionData(X, NodePositions, 100000, XSquared)

print(GraphGrammarOperation(
        X, NodePositions, ElasticMatrix, part, "removenode"))
