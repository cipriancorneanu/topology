import torch.nn as nn
import numpy as np 
from bisect import bisect
import array
import torch 
from graph import *


def get_operations(model):
    ''' Return all  operations of a model. Filter only linear and conv. '''
    return [m for m in model.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)]


def get_features(model, x):
    return model.forward_features(x)


def get_param_features(model, x):
    return model.forward_param_features(x)


def get_feature_sizes(model, x):
    ''' Return all feature sizes of a model. COMMENT: what about models with different input size? '''
    features =  model.forward_param_features(x)
    return [list(f.size()) for f in features]


def get_param_feature_sizes(model, x):
    ''' Return all feature sizes of a model. COMMENT: what about models with different input size? '''
    features =  model.forward_param_features(x)
    return [list(f.size()) for f in features]


def abs2rel(indices, sizes):
    ''' Given absolute indices, compute position and index relative to size '''
    cum = np.cumsum([0]+[np.prod(s) for s in sizes])
    return [(bisect(cum, x)-1, x-cum[bisect(cum, x)-1]) for x in indices]


def evaluate_group_node_importance(model, x, node_importance, random=False):
    ''' Group nodes and aggregate importance by filter'''
    g_nodes = group_nodes(model, x)
    g_importance = aggregate_importance(g_nodes, node_importance, method='mean')

    ''' Group node groups by and importance by layer '''
    n_filters = np.cumsum([0]+[x[1] for x in get_feature_sizes(model, x)])
    g_nodes = [g_nodes[beg:end] for beg, end in zip(n_filters[:-1], n_filters[1:])][:-1]
    g_importance = [g_importance[beg:end] for beg, end in zip(n_filters[:-1], n_filters[1:])][:-1]

    ''' Sort node groups for each layer '''
    ham = [([gn[x] for x in np.argsort(gi)], [gi[x] for x in np.argsort(gi)]) for gn, gi in zip(g_nodes, g_importance)]
    g_nodes = [x[0] for x in ham]
    min_n_filters = int(np.min([len(x[1]) for x in ham]))

    ''' Importance of filter for each layer is relative importance in layer '''
    g_importance = [list(range(len(x[1]))) for x in ham]

    ''' '''
    g_importance = [[int(item/(int(len(gi)/min_n_filters))) for item in gi] for gi in g_importance]
    
    ''' If random shuffle importance for each layer '''
    if random:
        g_importance = [np.random.permutation(gi) for gi in g_importance]

    ''' Linearize group nodes and correspoding importance '''
    g_nodes = [item for sublist in g_nodes for item in sublist]
    g_importance = [item for sublist in g_importance for item in sublist]

    return sorted([(x, y) for x, y in zip(g_nodes, g_importance)], key=lambda x: x[1])


def aggregate_importance(grouped_nodes, node_importance, method='mean'):
    if method=='mean':
        a_importance = [int(np.mean([node_importance[x] for x in g])) for g in grouped_nodes]
    if method=='median':           
        a_importance = [int(np.median([node_importance[x] for x in g])) for g in grouped_nodes]
    elif method=='max':
        a_importance = [int(np.max([node_importance[x] for x in g])) for g in grouped_nodes]
    elif method=='std':
        a_importance = [np.std([node_importance[x] for x in g]) for g in grouped_nodes]
    elif method=='nothing':
        a_importance = [[node_importance[x] for x in g] for g in grouped_nodes]
    return a_importance


def group_nodes(model, x):
    ops = get_operations(model)
    f_size = get_feature_sizes(model, x)
    cum_f_size = np.cumsum([0] + [np.prod(s) for s in f_size])
    
    grouped_nodes = []  
    for o, fs, fe in zip(ops, cum_f_size[:-1], cum_f_size[1:]):
        if isinstance(o, nn.Conv2d):
            s = [x for x in range(fs, fe, int((fe-fs)/o.out_channels))] + [fe]
            for start, end in zip(s[:-1], s[1:]):
                grouped_nodes.append([ x for x in range(start, end)] )
        else:
            for item in [x for x in range(fs, fe)]:
                grouped_nodes.append([item])
             
    return grouped_nodes
    
    
def get_mask(net, x, nodes):
    ''' Given a list of absolute node indices return a binary mask for all feature list. '''
    ops, feat_sizes = get_operations(net), get_feature_sizes(net, x)
    rel_nodes = abs2rel(nodes, feat_sizes)

    ''' Create mask '''
    mask = [torch.ones_like(f) for f in net.forward_param_features(x)]
    
    for (i_feat, i_node) in rel_nodes:        
        if i_feat < len(ops)-1: 
            unr = np.unravel_index(i_node, tuple(mask[i_feat].size()))
            mask[i_feat][unr] = 0
        
    return mask


def read_bin(fname):
    ''' Read binary file for adjacency matrix '''
    header = array.array("L")
    values = array.array("d")

    with open(fname, mode='rb') as file: # b is important -> binary
        header.fromfile(file, 3)
        values.fromfile(file, int(header[2]*header[2]))
        values = list(values)

        values = np.asarray([float("{0:.2f}".format(1-x)) for x in np.asarray(values)])
        values = np.reshape(values, (header[2], header[2]))

    return values


def evaluate_node_importance(adj, epsilon):
    ''' Evaluate node importance based on adjacency matrix and max cavity epsilon '''
    node_importance = np.zeros(adj.shape[0])
    adj[adj<epsilon]=0
    adj[adj>=epsilon]=1
    importance = np.sum(adj, axis=0)
    
    return (np.argsort(importance)[::-1], importance[np.argsort(importance)[::-1]])


def get_epc_eps(net, dataset, trl):
    if net == 'vgg' and dataset == 'cifar10':
        if trl in [23]:
            epc, eps = 130, 0.46
    if net == 'vgg' and dataset == 'svhn':
        if trl == 0:
            epc, eps = 40, 0.48
    if net == 'conv_2' and dataset == 'cifar10_gray28' and trl==0:
        epc, eps = 60, 0.29
    if net == 'conv_4' and dataset == 'cifar10_gray28' and trl==0:
        epc, eps = 60, 0.21
    if net == 'conv_6' and dataset == 'cifar10_gray28' and trl==0:
        epc, eps = 60, 0.29
    if net == 'conv_2' and dataset == 'svhn_gray28' and trl==0:
        epc, eps = 6, 0.20
    if net == 'conv_4' and dataset == 'svhn_gray28' and trl==0:
        epc, eps = 10, 0.20
    if net == 'conv_6' and dataset == 'svhn_gray28' and trl==0:
        epc, eps = 8, 0.20
    if net == 'lenet' and dataset == 'mnist':
        if trl in [0,1,2]:
            epc, eps = 10, 0.76
    elif net == 'lenet' and dataset == 'cifar10_gray28':
        if trl == 0:
            epc, eps = 10, 0.80
        if trl == 1:
            epc, eps = 16, 0.78
        if trl == 2:
            epc, eps = 6, 0.72
    elif net == 'lenet' and dataset == 'svhn_gray28':
        if trl == 0:
            epc, eps = 36, 0.64
        if trl == 1:
            epc, eps = 36, 0.64
        if trl == 2:
            epc, eps = 50, 0.60
    elif net == 'lenet' and dataset == 'fashion_mnist':
        print('Here')
        if trl == 0:
            epc, eps = 33, 0.66
        if trl == 1:
            epc, eps = 30, 0.64
        if trl == 2:
            epc, eps = 30, 0.64
    if net == 'lenet_300_100' and dataset == 'mnist':
        epc, eps = dummy, 0.7
    if net == 'lenet_300_100' and dataset == 'fashion_mnist':
        epc, eps = dummy, 0.7
    if net == 'lenet_300_100' and dataset == 'svhn_gray28':
        epc, eps = dummy, 0.7
    if net == 'lenet_300_100' and dataset == 'cifar10_gray28':
        epc, eps = dummy, 0.7

    return epc, eps


def compute_node_importance_adj(net, passer, eps, x):
    ops = get_operations(net)
    f_sizes = get_param_feature_sizes(net, x)

    print('Total number of nodes is {}'.format(np.sum([np.prod(x) for x in f_sizes])))    
    activs = passer.get_function(forward='parametric')
    activs = signal_concat(activs)

    sactivs = np.array_split(activs, 270, axis=0)
    nodes, importance, offset = [], [], 0
    sosobig = []
    for i_x, x in enumerate(sactivs):
        soso = []
        print('Computing slice {}/{}'.format(i_x, len(sactivs)))
        for i_y, y in enumerate(sactivs):
            adj = np.abs(np.nan_to_num(np.corrcoef(x, y)[:x.shape[0], x.shape[0]:]))
            '''adj = np.corrcoef(x, y)[:x.shape[0],x.shape[0]:]'''
            adj[adj<eps]=0
            adj[adj>=eps]=1
            imp = np.sum(adj, axis=1)
            nds = list(range(offset, offset+x.shape[0]))
            soso.append((nds, imp))

        sosobig.append(soso)
        offset = offset + x.shape[0]

    nodes, importance = [], []    
    for soso in sosobig:
        x = soso[0][1]
        for item in soso:
            x = x + item[1]

        nodes = nodes + soso[0][0]
        importance = importance + list(x)

    importance  = np.asarray(importance)

    return (np.argsort(importance)[::-1], importance[np.argsort(importance)[::-1]])

    
def compute_node_importance(net, dataset, trl):    
    epc, eps = get_epc_eps(net, dataset, trl)
    root = '/data/data1/datasets/cvpr2019/adjacency/'
    fname = root+'/'+net+'_'+dataset+'/adj_epc'+str(epc)+'_trl'+str(trl)+'.bin'
    adj = read_bin(fname)

    return evaluate_node_importance(adj, epsilon=eps)


def test():
    fname = '/data/data1/datasets/cvpr2019/adjacency/lenet_mnist/adj_epc10_trl0.bin'
    adj = read_bin(fname)
    nodes, importance = evaluate_node_importance(adj, epsilon=0.76)
    ops, struct_nodes = get_structure(model.module)

'''test()'''
