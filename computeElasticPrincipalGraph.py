# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:56:58 2018

@author: Alexis Martin
"""
import numpy as np
from ElPrincGraph import ElPrincGraph
from MakeUniformElasticMatrix import MakeUniformElasticMatrix


def pca(data):
    # computing eigenvalues and eigenvectors of covariance matrix
    mu = data.mean(axis=0)
    data = data - mu
    eigenvectors, eigenvalues, V = np.linalg.svd(data.T)
    return eigenvectors, V.T, eigenvalues


def computeElasticPrincipalGraph(data, NumNodes):
    return 0


X = np.array([[0, 1]])
with open("test_data/tree23.data") as F:
    for line in F:
        X = np.vstack((X, (np.array(line.split('\t')))[:2]))
X = X[0:56].astype(float)
coeff, score, latent = pca(X)
print(score)
