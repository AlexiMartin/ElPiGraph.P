# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:56:58 2018

@author: Alexis Martin
"""
import numpy as np
from core_algorithm.ElPrincGraph import ElPrincGraph
from core_algorithm.MakeUniformElasticMatrix import MakeUniformElasticMatrix
from scipy import linalg as la


def PCA(data):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return evecs, np.dot(evecs.T, data.T).T, evals


# TODO : graphs
def computeElasticPrincipalGraph(data, NumNodes, newDim=None, drawPCAview=True,
                                 drawAccuracyComplexity=True, drawEnergy=True,
                                 Lambda=0.01, Mu=0.1, InitNodeP=None,
                                 InitEdges=None, growGrammar=None,
                                 shrinkGrammar=None, ComputeMSEP=False,
                                 MaxBlockSize=100000, TrimmingRadius=np.inf,
                                 MaxNumberOfIterations=10, eps=0.01,
                                 verbose=True):
    NodeP = None
    EM = None
    if InitEdges is not None and InitNodeP is not None:
        NodeP = InitNodeP
        EM = MakeUniformElasticMatrix(InitEdges, Lambda, Mu)
    mv = data.mean(axis=0)
    data_centered = data - mv
    vglobal, uglobal, explainedVariances = PCA(data)
    if newDim is not None:
        tmp = newDim.shape[0]
        if tmp == 1:
            indPC = np.arange(newDim[0])
        elif tmp == 2:
            indPC = np.arange(newDim[0], newDim[1])
        else:
            indPC = newDim
        perc = explainedVariances[indPC].sum()/explainedVariances.sum()*100
        print("Varaince retained in ", indPC.shape[0], " : ", perc)
        data_centered = uglobal[:, indPC]
    else:
        indPC = np.arange(data.shape[1])

    NodePositions, ElasticMatrix = (
        ElPrincGraph(data_centered, NumNodes, Lambda, Mu, NodeP, EM,
                     growGrammar, shrinkGrammar, ComputeMSEP, MaxBlockSize,
                     TrimmingRadius, MaxNumberOfIterations, eps, verbose))

    if newDim is not None:
        NodePositions = np.dot(NodePositions, vglobal[:, indPC].T)
    # Edges = np.vstack(np.triu(ElasticMatrix, 1).nonzero())
    NodePositions += mv
    return NodePositions, ElasticMatrix
