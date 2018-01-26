# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:26:31 2018

@author:
"""
import numpy as np


# This is the core function for application of graph grammar approach for
# constructing primitive elastic principal graphs
#
# The function takes a definition of the principal graph embeddment and
# applies a graph grammar operation of type type
#
# Input arguments:
#   NodePositions - position of nodes of the primitive elastic graph
#   ElasticMatrix - matrix with elasticity coefficients (lambdas for edges,
#       and star elasticities along the diagonal
#   X - is dataset which is to be approximated by the graph
#   type - one of the operation types:
#       'addnode2node'      adds a node to each graph node
#       'removenode'        removes terminal node
#       'bisectedge'        adds nodt to the middle of each edge
#       'shrinkedge'        removes edge and glues two ends of this edge
#   partition is n-by-1 vector. partition(i) is number of node which is
#       associated with data point X(i,:).
#
# Outputs:
#   NodePositionArray - 3d array with dimensions
#       [node_number, NodePosition, graph_number], represents all generated
#       node configurations.
#   ElasticMatrices - 3d array with dimensions
#       [node_number, node_number, graph_number], represents all generated
#       elasticity matrices
#   NodeIndicesArray in the version 1.1 with a possibility of local search,
#       each operation reports NodeIndices which specifies how the nodes in
#       the newly generated graph are related to the nodes in the initial
#       graph, and contains zeros for new nodes.
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


# This grammar operation adds a node to each graph node
# The positions of the node is chosen as a linear extrapolation for a leaf
# node (in this case the elasticity of a newborn star is chosed as in
# BisectEdge operation),
# or
# as the data point giving the minimum local MSE for a star (without any
# optimization).
# TODO add pointweights ?
def AddNode2Node(X, NodePositions, ElasticMatrix, partition):
    nNodes = NodePositions.shape[0]
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    indL = Lambda > 0
    Connectivities = indL.sum(axis=0)
    # add pointweigths here if added
    assoc = (np.bincount(partition.ravel(), minlength=nNodes))
    # Create prototypes for new NodePositions, ElasticMatrix and inds
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack((np.hstack((ElasticMatrix, np.zeros((nNodes, 1)))),
                        np.zeros((1, nNodes+1))))
    niProt = np.arange(nNodes+1)
    niProt[nNodes] = 0
    # Put prototypes to corresponding places
    NodePositionsArray = np.repeat(npProt[:, :, np.newaxis], nNodes, axis=2)
    ElasticMatrices = np.repeat(emProt[:, :, np.newaxis], nNodes, axis=2)
    NodeIndicesArray = np.repeat(niProt[:, np.newaxis], nNodes, axis=1)

    for i in range(nNodes):
        # Compute mean edge elastisity for edges with node i
        meanL = Lambda[i, indL[i, ]].mean(axis=0)
        # Add edge to elasticity matrix
        ElasticMatrices[nNodes, i, i] = ElasticMatrices[i, nNodes, i] = meanL
        if Connectivities[i] == 1:
            # Add node to terminal node
            ineighbour = np.nonzero(indL[i, ])[0]
            # Calculate new node position
            NewNodePosition = 2*NodePositions[i, ]-NodePositions[ineighbour, ]
            # Complete Elasticity Matrix
            ElasticMatrices[i, i, i] = Mus[ineighbour]
        else:
            # Add node to a star
            # if 0 data points associated with this star
            if assoc[i] == 0:
                # then select mean of all leaves as new position
                NewNodePosition = NodePositions[indL[:, i]].mean(axis=0)

            else:
                # Otherwise take the mean of the points associated with the
                # central node
                NewNodePosition = X[(partition == i).ravel()].mean(axis=0)
        # fill node position
        NodePositionsArray[nNodes, :, i] = NewNodePosition
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray


# This grammar operation removes a leaf node (connectivity==1)
def RemoveNode(NodePositions, ElasticMatrix):
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    # Define sizes
    nNodes = ElasticMatrix.shape[0]
    nGraphs = (Connectivities == 1).sum()
    # Preallocate arrays
    NodePositionsArray = np.zeros((nNodes-1, NodePositions.shape[1], nGraphs))
    ElasticMatrices = np.zeros((nNodes-1, nNodes-1, nGraphs))
    NodeIndicesArray = np.zeros((nNodes-1, nGraphs))
    k = 0
    for i in range(Connectivities.shape[0]):
        if Connectivities[i] == 1:
            # if terminal node remove it
            newInds = np.concatenate((np.arange(0, i), np.arange(i+1, nNodes)))
            NodePositionsArray[:, :, k] = NodePositions[newInds, :]
            tmp = np.repeat(False, nNodes)
            tmp[newInds] = True
            tmp2 = ElasticMatrix[tmp, :]
            ElasticMatrices[:, :, k] = tmp2[:, tmp]
            NodeIndicesArray[:, k] = newInds
            k += 1
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray


# This grammar operation inserts a node inside the middle of each edge
# The elasticity of the edges do not change
# The elasticity of the newborn star is chosen as
# mean over the neighbour stars if the edge connects two star centers
# or
# the one of the single neigbour star if this is a dangling edge
# or
# if one starts from a single edge, the star elasticities should be on
# one of two elements in the diagoal of the ElasticMatrix
def BisectEdge(NodePositions, ElasticMatrix):
    # Decompose Elastic Matrix: Mus
    Mus = ElasticMatrix.diagonal()
    # Get list of edges
    start, stop = np.triu(ElasticMatrix, 1).nonzero()
    # Define some constants
    nGraphs = start.shape[0]
    nNodes = NodePositions.shape[0]
    # Create prototypes for new NodePositions, ElasticMatrix and inds
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack((np.hstack((ElasticMatrix, np.zeros((nNodes, 1)))),
                        np.zeros((1, nNodes+1))))
    niProt = np.arange(nNodes+1)
    niProt[nNodes] = 0
    # Allocate arrays and put prototypes in place
    NodePositionsArray = np.repeat(npProt[:, :, np.newaxis], nGraphs, axis=2)
    ElasticMatrices = np.repeat(emProt[:, :, np.newaxis], nGraphs, axis=2)
    NodeIndicesArray = np.repeat(niProt[:, np.newaxis], nGraphs, axis=1)
    for i in range(nGraphs):
        NewNodePosition = (NodePositions[start[i], ] +
                           NodePositions[stop[i], ]) / 2
        # Fill NodePosition
        NodePositionsArray[nNodes, :, i] = NewNodePosition
        # correct elastic matrix
        Lambda = ElasticMatrix[start[i], stop[i]]
        # remove edge
        ElasticMatrices[start[i], stop[i], i] = 0
        ElasticMatrices[stop[i], start[i], i] = 0
        # add 2 edges
        ElasticMatrices[start[i], nNodes, i] = Lambda
        ElasticMatrices[nNodes, start[i], i] = Lambda
        ElasticMatrices[nNodes, stop[i], i] = Lambda
        ElasticMatrices[stop[i], nNodes, i] = Lambda
        # Define mus of edges
        mu1 = Mus[start[i]]
        mu2 = Mus[stop[i]]
        if mu1 > 0 and mu2 > 0:
            ElasticMatrices[nNodes, nNodes, i] = (mu1 + mu2) / 2
        else:
            ElasticMatrices[nNodes, nNodes, i] = max(mu1, mu2)
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray


# This grammar operation removes an edge from the graph
# If this is an edge connecting a leaf node then it is equivalent to
# RemoveNode. So we remove only internal edges.
# If this is an edge connecting two stars then their leaves are merged,
# and the star is placed in the middle of the shrinked edge.
# The elasticity of the new formed star is the average of two star
# elasticities.
def ShrinkEdge(NodePositions, ElasticMatrix):
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    # get list of edges
    start, stop = np.triu(ElasticMatrix, 1).nonzero()
    # define size
    nNodes = NodePositions.shape[0]
    # identify edges with minimal connectivity > 1
    ind = np.min(np.hstack((Connectivities[start[np.newaxis].T],
                            Connectivities[stop[np.newaxis].T])), axis=1)
    ind = ind > 1
    start = start[ind]
    stop = stop[ind]
    # calculate nb of graphs
    nGraphs = start.shape[0]
    # preallocate array
    NodePositionsArray = np.zeros((nNodes-1, NodePositions.shape[1], nGraphs))
    ElasticMatrices = np.zeros((nNodes-1, nNodes-1, nGraphs))
    NodeIndicesArray = np.zeros((nNodes-1, nGraphs))
    for i in range(stop.shape[0]):
        # create copy of elastic matrix
        em = ElasticMatrix.copy()
        # Reattaches all edges connected with stop[i] to start[i]
        # and make a new star with an elasticity average of two merged stars
        em[start[i], ] = np.maximum(Lambda[start[i], ], Lambda[stop[i], ])
        em[:, start[i]] = np.maximum(Lambda[:, start[i]], Lambda[:, stop[i]])
        em[start[i], start[i]] = (Mus[start[i]] + Mus[stop[i]]) / 2
        # Create copy of node positions
        nodep = NodePositions.copy()
        # madify node start[i]
        nodep[start[i], :] = (nodep[start[i], :] + nodep[stop[i], :]) / 2
        # Form index for retained nodes and extract corresponding part of
        # node positions and elastic matrix
        newInds = np.concatenate((np.arange(0, stop[i]),
                                 np.arange(stop[i]+1, nNodes)))
        NodePositionsArray[:, :, i] = nodep[newInds, ]
        ElasticMatrices[:, :, i] = em.take(newInds, axis=0).take(newInds,
                                                                 axis=1)
        NodeIndicesArray[:, i] = newInds
    return NodePositionsArray, ElasticMatrices, NodeIndicesArray
