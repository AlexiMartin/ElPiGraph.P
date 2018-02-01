# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:48:28 2018

@author: Alexis Martin
"""
import numpy as np
from core_algorithm.MakeUniformElasticMatrix import MakeUniformElasticMatrix
import core_algorithm.ApplyOptimalGraphGrammarOperation as ao


# TODO add report
def ElPrincGraph(X, NumNodes, Lambda, Mu, InitNodePosition=None,
                 InitElasticMatrix=None, growGrammar=None, shrinkGrammar=None,
                 ComputeMSEP=False, MaxBlockSize=100000,
                 TrimmingRadius=np.inf, MaxNumberOfIterations=10,
                 eps=0.01, verbose=True):
    if growGrammar is None:
        growGrammar = np.array([["bisectedge", "addnode2node"],
                                ["bisectedge", "addnode2node"]])
    if shrinkGrammar is None:
        shrinkGrammar = np.array([["shrinkedge", "removenode"]])
    NodeP = InitNodePosition
    em = InitElasticMatrix
    if em is not None and np.all(em != em.T):
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
    if verbose:
        print('BARCODE\tENERGY\tNNODES\tNEDGES\tNRIBS\tNSTARS' +
              '\tNRAYS\tNRAYS2\tMSE MSEP\tFVE\tFVEP\tUE\tUR\tURN\tURN2\tURSD')
    if growGrammar.shape[0] <= shrinkGrammar.shape[0]:
        raise ValueError("The tree cannot grow if less growing grammar than " +
                         "shrinking grammar.")
    while NodeP.shape[0] < NumNodes:
        for k in range(growGrammar.shape[0]):
            NodeP, em, partition, dists = (
                    ao.ApplyOptimalGraphGrammarOperation(X, NodeP, em,
                                                         growGrammar[k],
                                                         MaxBlockSize, verbose,
                                                         TrimmingRadius,
                                                         MaxNumberOfIterations,
                                                         eps))
        for k in range(shrinkGrammar.shape[0]):
            NodeP, em, partition, dists = (
                    ao.ApplyOptimalGraphGrammarOperation(X, NodeP, em,
                                                         shrinkGrammar[k],
                                                         MaxBlockSize, verbose,
                                                         TrimmingRadius,
                                                         MaxNumberOfIterations,
                                                         eps))
    return NodeP, em
