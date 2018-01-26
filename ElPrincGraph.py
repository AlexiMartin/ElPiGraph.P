# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:48:28 2018

@author: Alexis Martin
"""
import numpy as np
from MakeUniformElasticMatrix import MakeUniformElasticMatrix
from ApplyOptimalGraphGrammarOperation import ApplyOptimalGraphGrammarOperation


def ElPrincGraph(X, NumNodes, Lambda, Mu, InitNodePosition=None,
                 InitElasticMatrix=None, growGrammar=None, shrinkGrammar=None,
                 ComputeMSEP=False, MaxBlockSize=100000,
                 TrimmingRadius=np.inf, MaxNumberOfIterations=10,
                 eps=0.01, verbose=True, report=True):
    if growGrammar is None:
        growGrammar = np.array([["bisectedge", "addnode2node"],
                                ["bisectedge", "addnode2node"]])
    print(growGrammar[0])
    if shrinkGrammar is None:
        shrinkGrammar = np.array([["shrinkedge", "removenode"]])
    NodeP = InitNodePosition
    em = InitElasticMatrix
    if em is not None and em != em.T:
        raise ValueError("Elastic Matrix must be square and symmetric")
    if NodeP is not None:
        CurrentNumberOfNodes = NodeP.shape[0]
    else:
        CurrentNumberOfNodes = 0
    if em is None:
        if CurrentNumberOfNodes == 0:
            edges = np.array([[0, 1]])
        else:
            edges = np.vstack((np.arange(CurrentNumberOfNodes-1),
                               np.arange(1, CurrentNumberOfNodes))).T
        em = MakeUniformElasticMatrix(edges, Lambda, Mu)
    if CurrentNumberOfNodes == 0:
        CurrentNumberOfNodes = em.shape[0]
        _, _, v = np.linalg.svd(X)
        v = abs(v[0, ])
        mn = X.mean(axis=0)
        st = np.std((X*v).sum(axis=1), ddof=1)
        delta = 2 * st / (CurrentNumberOfNodes - 1)
        NodeP = ((mn-st*v) +
                 ((delta * range(CurrentNumberOfNodes))[np.newaxis].T * v))
    CurrentNumberOfNodes = NodeP.shape[0]
    UR = em.diagonal()
    if (UR > 0).sum() == 0:
        em = em + np.diag(Mu*np.ones((CurrentNumberOfNodes)))
    print(em)
    if verbose:
        print('BARCODE\tENERGY\tNNODES\tNEDGES\tNRIBS\tNSTARS' +
              '\tNRAYS\tNRAYS2\tMSE MSEP\tFVE\tFVEP\tUE\tUR\tURN\tURN2\tURSD')
    if report:
        BARCODES = '';
        ENERGY = np.zeros((NumNodes - CurrentNumberOfNodes, 1)); 
        NNODES = ENERGY; 
        NEDGES = ENERGY; 
        NRIBS = ENERGY; 
        NSTARS = ENERGY; 
        NRAYS = ENERGY; 
        NRAYS2 = ENERGY; 
        MSE = ENERGY; 
        MSEP = ENERGY; 
        FVE = ENERGY; 
        FVEP = ENERGY; 
        UE = ENERGY; 
        UR = ENERGY; 
        URN = ENERGY; 
        URN2 = ENERGY; 
        URSD = ENERGY;
    i=0
    while NodeP.shape[0] < NumNodes:
        for k in range(growGrammar.shape[1]):
            NodeP, em, partition, dists = (
                    ApplyOptimalGraphGrammarOperation(X, NodeP, em,
                                                      growGrammar[k],
                                                      MaxBlockSize, verbose,
                                                      TrimmingRadius,
                                                      MaxNumberOfIterations,
                                                      eps))
        for k in range(shrinkGrammar.shape[1]):
            NodeP, em, partition, dists = (
                    ApplyOptimalGraphGrammarOperation(X, NodeP, em,
                                                      shrinkGrammar[k],
                                                      MaxBlockSize, verbose,
                                                      TrimmingRadius,
                                                      MaxNumberOfIterations,
                                                      eps))
        if report:
            ReportOnPrimitiveGraphEmbedment(X, np, em, partition, dists,
                                            ComputeMSEP)


ElPrincGraph(np.array([[0.88129, 0.67394], [0.55545, 0.58511], [0.61349, 0.77580], [0.24102, 0.92581], [0.55211, 0.10624]]), 1, 1, 2)
# np.array([[0.88129, 0.67394, 0.51773], [0.55545, 0.58511, 0.43339], [0.61349, 0.77580, 0.62440], [0.24102, 0.92581, 0.93493], [0.55211, 0.10624, 0.91211]])  
