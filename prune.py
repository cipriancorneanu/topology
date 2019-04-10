
import torch.nn as nn
import numpy as np 
from bisect import bisect




def abs2rel_nodes(model, nodes):
    ''' Get relative node index to layer (TO ADD conv) '''
    n_nodes = np.hstack((np.asarray([0]), np.cumsum([m.weight.data.shape[0] for m in model.modules() if isinstance(m, nn.Linear)])))
    return [(bisect(n_nodes, node)-1, node-n_nodes[bisect(n_nodes, node)-1]) for node in nodes]


def prune_nodes(model, nodes):
    ''' Prune nodes in model '''
    rel_nodes = abs2rel_nodes(model, nodes)
    modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    
    for node in rel_nodes:
        ''' Don't touch last layer nodes! '''
        if node[0]!=2:
            modules[node[0]].weight.data[node[1], :] = 0
            modules[node[0]].weight.requires_grad = False                                                   

    return model


def read_betti(fname):
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
    return  np.argsort(importance)[::-1]


def test():
    fname = './activations/adj_epc15_trl0.bin'
    adj = read_betti(fname, dimension=1, persistence=0.03)
    node_importance = evaluate_node_importance(adj, epsilon=0.7)

    print(node_importance)

'''test()'''
