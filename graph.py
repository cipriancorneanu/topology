__author__ = 'cipriancorneanu'
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import argparse
from sklearn.decomposition import PCA
import h5py
import os
import pickle as pkl
import scipy.stats


def correlation(x, y):
    return np.corrcoef(x,y)[0,1]

def kl(x, y):
    ''' 
    Return Kullback-Leibler divergence btw two probability density functions
    x, y: 1D nd array, sum(x)=1, sum(y)=1
    '''
    return scipy.stats.entropy(x, y)

    
def corrpdf(signals):
    '''
    Compute pdf of correlations between signals
    signals: 2D ndarray, each row is a signal 
    '''
    
    ''' Get correlation matrix '''
    x = np.corrcoef(signals)

    ''' Get upper triangular part and vectorize'''
    x = np.triu(x).flatten()
    ''' Beware of diagonal and zeros !!'''
    
    ''' Compute pdf'''
    pdf, _ = np.histogram(x, bins=np.arange(-1, 1, 0.01), density=True)
    
    return pdf

def adjacency_correlation(signals):
    ''' Faster version of adjacency matrix with correlation metric '''
    signals = np.reshape(signals, (signals.shape[0], -1))
    return np.nan_to_num(np.corrcoef(signals))
    
def adjacency(signals, metric):
    '''
    Build matrix A  of dimensions nxn where a_{ij} = metric(a_i, a_j).
    signals: nxm matrix whre each row (signal[k], k=range(n)) is a signal. 
    metric: a function f(.,.) that takes two 1D ndarrays and outputs a single real number (e.g correlation, KL divergence etc).
    '''
    
    ''' Get input dimensions '''
    signals = np.reshape(signals, (signals.shape[0], -1))
    n, m = signals.shape

    A = np.zeros((n, n))

    print(signals[2].shape)

    for i in range(n):
        for j in range(n):
            A[i,j] = metric(signals[i], np.transpose(signals[j]))
            
    return A
    
