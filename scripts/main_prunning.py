import argparse
import os
import numpy as np 
from config import SAVE_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--build_graph', default=1, type=int)
parser.add_argument('--net', help='Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)')
parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
parser.add_argument('--trial', default=0, help='Specify trial number. Used to differentiate btw multiple trainings of same setup.')
parser.add_argument('--n_epochs_train', default='10', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.01', help='Specify learnig rate for training.')
parser.add_argument('--permute_labels', default='0.0', help='Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels.')
parser.add_argument('--binarize_labels', default='-1', help='If positive, Binarize labels. Put label equal to binarize_labels to 1. All the rest put to zero.')
parser.add_argument('--data_subset', default='1.0', help='Specify if subset of data should be loaded. Float between 0 and 1. If 0, all data, else proportion of data randomly sampled.')
parser.add_argument('--epochs_test', help='Epochs for which you want to build graph. String of positive natural numbers separated by spaces.')
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start step stop \' ')
parser.add_argument('--filtration', default='nominal')
parser.add_argument('--scale', default='linear')
parser.add_argument('--split',  default='0', help='Split network into partitions.')
parser.add_argument('--subsplit',  default='0', help='Split each partition.')
parser.add_argument('--kl', default='0', help='TO ADD.')
parser.add_argument('--graph_type', default='functional')
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--select_nodes', default='0')
parser.add_argument('--partition', default='hardcoded')
parser.add_argument('--homology_type', default='static')
args = parser.parse_args()

''' Create thresholds argument (logarithmic) '''
start, stop = tuple([float(x) for x in args.thresholds.split()])
if args.scale == 'logarithmic':
    thresholds = str(['{:.4f}'.format(x) for x in  np.geomspace(start, stop)])[1:-1].replace('\'','').replace(',','')
elif args.scale == 'linear':
    step = 0.025
    thresholds = str(['{:.4f}'.format(x) for x in np.arange(start, stop, step)])[1:-1].replace('\'','').replace(',','')
    
print(thresholds)

def visible_print(message):
    ''' Visible print'''
    print('')
    print(50*'-')
    print(message)
    print(50*'-')
    print('')

    
if args.train:
    visible_print('Training network')
    os.system('python ../train.py --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.n_epochs_train+' --lr '+args.lr+' --permute_labels '+args.permute_labels+' --subset '+args.data_subset+' --binarize_labels '+args.binarize_labels)
