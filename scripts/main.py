import argparse
import os
import numpy as np 
from config import SAVE_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--net', help='Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)')
parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
parser.add_argument('--trial', default=0, help='Specify trial number. Used to differentiate btw multiple trainings of same setup.')
parser.add_argument('--n_epochs_train', default='10', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.01', help='Specify learnig rate for training.')
parser.add_argument('--permute_labels', default='0.0', help='Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels.')
parser.add_argument('--binarize_labels', default='-1', help='If positive, Binarize labels. Put label equal to binarize_labels to 1. All the rest put to zero.')
parser.add_argument('--data_subset', default='1.0', help='Specify if subset of data should be loaded. Float between 0 and 1. If 0, all data, else proportion of data randomly sampled.')
parser.add_argument('--epochs_test', help='Epochs for which you want to build graph. String of positive natural numbers separated by spaces.')
parser.add_argument('--thresholds', help='Defining thresholds range in the form \'start step stop \' ')
parser.add_argument('--split',  default='0', help='Split layers into chunks.')
parser.add_argument('--kl', default='0', help='TO ADD.')
parser.add_argument('--graph_type', default='functional')
parser.add_argument('--n_samples', type=int, default=5)
args = parser.parse_args()

''' Create thresholds argument '''
start, step, stop = tuple([float(x) for x in args.thresholds.split()])
thresholds = str(np.arange(start, stop, step))[1:-1].replace('\n', ' ')

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

visible_print('Building '+args.graph_type+' graph')
if args.graph_type=='functional':
    os.system('python ../build_graph_functional.py --save_path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --split '+args.split+' --kl '+args.kl+' --permute_labels '+args.permute_labels+' --binarize_labels '+args.binarize_labels)
elif args.graph_type=='functional_bn':
    os.system('python ../build_graph_functional_bn.py --save_path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --split '+args.split+' --kl '+args.kl+' --permute_labels '+args.permute_labels+' --binarize_labels '+args.binarize_labels)
elif args.graph_type=='functional_persample':
    os.system('python ../build_graph_functional_persample.py --save_path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --permute_labels '+args.permute_labels+' --binarize_labels '+args.binarize_labels+' --n_samples '+str(args.n_samples))

    
visible_print('Computing topology')
if args.graph_type=='functional_persample':
    for sample in range(args.n_samples):
        os.system('python compute_topology_persample.py --save_path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --trial '+ args.trial + ' --sample '+str(sample))
else:
    os.system('python compute_topology.py --save_path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --trial '+ args.trial)

visible_print('Prepare topology results')
if args.graph_type=='functional_per_sample':
    os.system('python prepare_results.py --path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --permute_labels '+args.permute_labels+' --subset '+args.data_subset)
else:
    for sample in range(args.n_samples):
        os.system('python prepare_results_persample.py --path '+SAVE_PATH+' --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.epochs_test+' --thresholds '+thresholds+' --permute_labels '+args.permute_labels+' --subset '+args.data_subset +' --sample ' + str(sample))
