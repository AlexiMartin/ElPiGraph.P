# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:11:02 2018

@author: Alexis Martin
"""
import numpy as np
from scipy import linalg as la
from core_algorithm.PartitionData import PartitionData


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


def PCAView(Nodes, Edges, data, pc1=np.array([1]), pc2=np.array([2]), pc1FVE=0,
            pc2FVE=0, TrimmingRadius=np.inf):
    TrimmingRadius *= TrimmingRadius
    data_centered = data - data.mean(axis=0)
    Nodes_centered = Nodes - data.mean(axis=0)
    if pc1.shape[0] == 1 and pc2 == np.array([2]):
        pc2 = pc1+1
    else:
        pc2 = 1
    if pc1.shape[0] == 1 or pc2.shape[0] == 1:
        vglobal, uglobal, explainedVariances = PCA(data_centered)
    if pc1.shape[0] == 1:
        xData = uglobal[:, pc1]
        xNodes = Nodes_centered * vglobal[:, pc1]
        pc1FVE = explainedVariances[pc1] / explainedVariances.sum()
    else:
        xData = data_centered * pc1
        xNodes = Nodes_centered * pc1
    if pc2.shape[0] == 1:
        ydata = uglobal[:, pc2]
        yNodes = Nodes_centered * vglobal[:, pc2]
        pc2FVE = explainedVariances[pc2] / explainedVariances.sum()
    else:
        yData = data_centered * pc2
        yNodes = Nodes_centered * pc2
    SquaredX = (data_centered**2).sum(axis=1).reshape((data_centered.shape[0], 1))
    partition = PartitionData(data, Nodes, 10000, SquaredX, TrimmingRadius)[0]
    Nodes_size = np.bincount(partition+1, minlength=NumberOfNodes+1)
    Nodes_size = Nodes_size[1:]
        
