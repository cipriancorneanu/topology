import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--thresholds', nargs='+')
parser.add_argument('--n_bettis', type=int, default=3)
parser.add_argument('--permute_labels', type=float)
parser.add_argument('--subset', type=float)
args = parser.parse_args()

PATH = '/data/data1/datasets/cvpr2019/adjacency/' + args.net + '_' + args.dataset + '/'

bettis = {}
for e in args.epochs:
    epoch = {}
    for t in args.thresholds:
        
        """ Read bettis from correspoding file """
        fname = PATH + 'badj_epc{}_t{}_trl{}.csv_symmetric_bettis.txt'.format(e, t, args.trial)

        with open(fname) as f:
            content = f.readlines()

        content = content[0].split(',')[1:1+args.n_bettis]
        content = [int(x) for x in content]

        
        epoch['t_{}'.format(t)] = content

    bettis['epc_{}'.format(e)] = epoch

''' Save bettis '''
with open(PATH + 'bettis_p{}_s{}.pkl'.format(args.permute_labels, args.subset), "wb") as f:
    pickle.dump(bettis, f, pickle.HIGHEST_PROTOCOL)
