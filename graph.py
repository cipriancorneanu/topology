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


parser = argparse.ArgumentParser()
parser.add_argument('--network', default='lenet')
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--threshold', type=float, default=.6)
parser.add_argument('--trial', type=int, default=0)
parser.add_argument('--start_layer', type=int, default=0)
parser.add_argument('--stop_layer', type=int, default=0)
parser.add_argument('--chunking', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)

args = parser.parse_args()


def correlation(x, y):
    return np.corrcoef(x,y)

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


def adjacency(signals, metric):
    '''
    Build matrix A  of dimensions nxn where a_{ij} = metric(a_i, a_j).
    signals: nxn matrix whre each row (signal[k], k=range(n)) is a signal. 
    metric: a function f(.,.) that takes two 1D ndarrays and outputs a single real number (e.g correlation, KL divergence etc).
 
    '''
    
    ''' Get input dimensions '''
    n, m = signals.shape()

    A = np.zeros(n, n)

    for i in range(n):
        for j in range(n):
            A[i,j] = metric(signals[i], signals[j])
            
    return A
    
    
def read_activations_pkl(fname):
    with open(fname, 'rb') as f:
        activations = pkl.load(f)
    return activations


def read_activations(fname, epoch):
    f = h5py.File(fname, 'r')
    layers = [l for l in list(f.get(epoch+'/activations/').keys())]
    slayers = sorted(layers, key = lambda x: int(x.split("_")[1]))
    print(slayers)
    
    return [np.array(f.get(epoch+'/activations/'+l)) for l in slayers]
        

def build_adjacency(activs, binarize_t=None):
    nodes = np.concatenate([np.transpose(a.reshape(a.shape[0], -1)) for a in activs], axis=0)

    print('Shape of the nodes is {}'.format(nodes.shape))
    
    ''' Compute correlation matrix '''
    corr = np.nan_to_num(np.corrcoef(nodes))
    
    ''' Binarize matrix '''
    if binarize_t:
        corr[corr>binarize_t] = 1
        corr[corr<=binarize_t] = 0
    
    density = np.sum(corr)/np.prod(corr.shape)
    
    return corr, density


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


def build_adjacency_subset_layerwise(activs,  sz_chunk, binarize_t=None):
    out = 512
    ''' Build adjacency matrix from node activations'''
    for i_a, a in enumerate(activs):
        sz_layer = np.prod(np.shape(a)[1:])
        
        if sz_layer >= sz_chunk: 
            nodes = np.transpose(a.reshape(a.shape[0], -1))
            subset = np.random.randint(0, nodes.shape[0], size=sz_chunk)
            nodes = nodes[subset, :]
            
            '''nodes = np.concatenate([np.transpose(a.reshape(a.shape[0], -1)) for a in activs], axis=0)'''
    
            ''' Compute correlation matrix '''
            corr = np.nan_to_num(np.corrcoef(nodes))
                
            ''' Binarize matrix '''
            if binarize_t:
                corr[corr>binarize_t] = 1
                corr[corr<=binarize_t] = 0

            out.append({'layer':i_a, 'chunk': 0, 'data': corr})

    return out


def build_adjacency_layerwise(activs,  sz_chunk, binarize_t=None):
    out = []
    ''' Build adjacency matrix from node activations '''
    for i_a, a in enumerate(activs):
        sz_layer = np.prod(np.shape(a)[1:])

        if sz_layer > sz_chunk:
            chunks = np.split(a, sz_layer/sz_chunk, axis=1)
        else:
            chunks = [a]
            
        for i_chunk, chunk in enumerate(chunks):
            nodes = np.transpose(chunk.reshape(chunk.shape[0], -1))
            '''nodes = np.concatenate([np.transpose(a.reshape(a.shape[0], -1)) for a in activs], axis=0)'''
    
            ''' Compute correlation matrix '''
            corr = np.nan_to_num(np.corrcoef(nodes))
                
            ''' Binarize matrix '''
            if binarize_t:
                corr[corr>binarize_t] = 1
                corr[corr<=binarize_t] = 0

            out.append({'layer':i_a, 'chunk': i_chunk, 'data': corr})

    return out
    


def build_batch_adjacency_layerwise(network, dataset, trial, epochs, thresholds, chunk_size=0):
    load_path = './activations/'+network+'_'+dataset+'/'
    start_layer=3 if network in ['vgg', 'resnet'] else 0

    for i_epc, epc in enumerate(epochs):
        for i_threshold, threshold in enumerate(thresholds):

            fname = load_path+'activations_trial_'+str(trial)+'.hdf5'                
            activs = read_activations(fname, 'epoch_{}'.format(epc))[start_layer:]

            if chunk_size:
                badj = build_adjacency_layerwise(activs, sz_chunk=chunk_size, binarize_t=threshold)

                for x in badj:
                    save_path = '/data/data1/datasets/cvpr2019/adjacency/'+network+'_'+dataset+'/{}/layer{}_chunk{}/'.format(chunk_size, start_layer + x['layer'], x['chunk'])
                                    
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    print('Saving ... trl{}, epc{}, t{:.3f}, layer{}, chunk{}, shape {}'.format(trial, epc, threshold, start_layer+x['layer'], x['chunk'], x['data'].shape))
                    np.savetxt(save_path+'badj_epc{}_trl{}_t{:.3f}.csv'.format(epc, trial, threshold), x['data'], fmt='%d', delimiter=",")
                    
            else:
                badj = build_adjacency(activs, binarize_t=threshold)
                save_path = '/data/datasets/cvpr2019/adjacency/'+network+'_'+dataset+'/'
                np.savetxt(save_path+'badj_epc{}_trl{}_t{:.3f}.csv'.format(epc, trial, threshold), badj, fmt='%d', delimiter=",")
                

def build_batch_adjacency(network, dataset, trial, epochs, thresholds, chunk_size=0):
    load_path = './activations/'+network+'_'+dataset+'/'

    start_layer=3 if network in ['vgg', 'resnet'] else 0
    
    for i_epc, epc in enumerate(epochs):
        for i_threshold, threshold in enumerate(thresholds):
            fname = load_path+'activations_trial_'+str(trial)+'.hdf5'
            print(fname)
            activs = read_activations(fname, 'epoch_{}'.format(epc))[start_layer:]
        
            badj, dens = build_adjacency(activs, binarize_t=threshold)
            print("Computing adjacency matrix of trial {}, epoch {}, threshold {} / density={}".format(trial, epc, threshold, dens))
            
            save_path = '/data/data1/datasets/cvpr2019/adjacency/'+network+'_'+dataset+'/'
            
            print('Saving file ' + save_path+'badj_epc{}_trl{}_t{:.3f}.csv'.format(epc, trial, threshold))
            np.savetxt(save_path+'badj_epc{}_trl{}_t{:.3f}.csv'.format(epc, trial, threshold), badj, fmt='%d', delimiter=",")

        
def build_batch_density_adjacency(network, dataset, trial, epochs, densities, chunk_size=0):
    load_path = './activations/'+network+'_'+dataset+'/'
    start_layer=3 if network in ['vgg', 'resnet'] else 0
    
    for i_epc, epc in enumerate(epochs):
        for i_density, density in enumerate(densities):
            fname = load_path+'activations_trial_'+str(trial)+'.hdf5'
            print(fname)
            activs = read_activations(fname, 'epoch_{}'.format(epc))[start_layer:]
        
            badj, dens, threshold = build_density_adjacency(activs, density_t=density)
            print("Computing adjacency matrix of trial {}, epoch {}, threshold {} / density={}".format(trial, epc, threshold, dens))
            
            save_path = '/data/data1/datasets/cvpr2019/adjacency/'+network+'_'+dataset+'/'
            print('Saving file ' + save_path+'badj_epc{}_trl{}_d{:.3f}.csv'.format(epc, trial, dens))
            np.savetxt(save_path+'badj_epc{}_trl{}_d{:.3f}.csv'.format(epc, trial, dens), badj, fmt='%d', delimiter=",")



def main():
    build_batch_adjacency_layerwise(network=args.network, dataset=args.dataset, trial=args.trial, epochs=[args.epoch], thresholds=[args.threshold], chunk_size=args.chunking)

'''
def main():
    build_batch_adjacency(network=args.network, dataset=args.dataset, trial=args.trial, epochs=[args.epoch], thresholds=[args.threshold])
'''

main()
