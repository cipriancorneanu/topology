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


def binarize(M, binarize_t):
    ''' Binarize matrix. Real subunitary values. '''
    M[M>binarize_t] = 1
    M[M<=binarize_t] = 0
    
    return M
    

def adjacency(signals, metric=None):
    '''
    Build matrix A  of dimensions nxn where a_{ij} = metric(a_i, a_j).
    signals: nxm matrix whre each row (signal[k], k=range(n)) is a signal. 
    metric: a function f(.,.) that takes two 1D ndarrays and outputs a single real number (e.g correlation, KL divergence etc).
    '''
    
    ''' Get input dimensions '''
    signals = np.reshape(signals, (signals.shape[0], -1))
    ''' If no metric provided fast-compute correlation  '''
    if not metric:
        return np.nan_to_num(np.corrcoef(signals))
        
    n, m = signals.shape

    A = np.zeros((n, n))

    print(signals[2].shape)

    for i in range(n):
        for j in range(n):
            A[i,j] = metric(signals[i], np.transpose(signals[j]))
            
    return A


def build_adjacency(signals, binarize_t=None):
    '''
    Build adjacency matrix
    '''
    
    ''' Compute adjacency (default metric is correlation)'''
    A = adjacency(signals)
    
    ''' Binarize '''
    if binarize_t:
        A = binarize(A, binarize_t)
    
    return A


def build_adjacency_split(signals,  sz_chunk, binarize_t=None):
    ''' 
    Split signals into chunks and Build adjacency matrix for each.
    signals: a list of signals. 
    sz_chunk: size of chunks to split signal 
    '''

    out = []
    for i_x, x in enumerate(signals):
        sz = np.prod(np.shape(x)[1:])
        
        if sz > sz_chunk:
            chunks = np.array_split(x, sz/sz_chunk, axis=1)
        else:
            chunks = [x]
            
        for i_chunk, chunk in enumerate(chunks):
            nodes = np.transpose(chunk.reshape(chunk.shape[0], -1))
            '''nodes = np.concatenate([np.transpose(a.reshape(a.shape[0], -1)) for a in activs], axis=0)'''

            A = adjacency(nodes)

            if binarize_t:
                A = binarize(A, binarize_t)
            
            out.append({'layer':i_x, 'chunk': i_chunk, 'data': A})

    return out

def build_density_adjacency(activs, density_t):
    nodes = np.concatenate([np.transpose(a.reshape(a.shape[0], -1)) for a in activs], axis=0)

    ''' Compute correlation matrix '''
    corr = np.nan_to_num(np.corrcoef(nodes))
    total_edges = np.prod(corr.shape)
    
    
    ''' Binarize matrix '''
    t, t_decr = 1, 0.005
    while True:
        ''' Decrease threshold until density is met '''
        edges = np.sum(corr > t)
        density = edges/total_edges
        ''' print('Threhold: {}; Density:{}'.format(t, density))'''
        
        if density > density_t:
            corr[corr > t] = 1
            corr[corr <= t] = 0
            break

        t = t-t_decr
        
    return corr, density, t
