import os
import argparse
import pickle
from bettis import *

parser = argparse.ArgumentParser()
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--permute_labels', type=float, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--subset', type=float, default=1)
parser.add_argument('--path')
args = parser.parse_args()

path = args.path + 'adjacency/' + args.net + '_' + args.dataset + '/'
thresholds = [float(x) for x in args.thresholds]

""" Compile bettis from correspoding file and prepare for saving """
bettis = {}
for e in args.epochs:
    epoch = {}
    for t in thresholds:
        fname = path + 'badj_epc{}_t{:1.2f}_trl{}.csv_symmetric_bettis.txt'.format(e, t, args.trial)
        epoch['t_{:1.2f}'.format(t)] = compile_bettis(fname, n_bettis=3)

    measure = betti_integral(epoch, min(thresholds), max(thresholds))
    bettis['epc_{}'.format(e)] = (epoch, measure)

''' Save bettis '''
with open(path + 'bettis_trl{}_p{}_s{}_split{}.pkl'.format(args.trial, args.permute_labels, args.subset, args.split), "wb") as f:
    pickle.dump(bettis, f, pickle.HIGHEST_PROTOCOL)
