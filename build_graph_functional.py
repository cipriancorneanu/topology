from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
from utils import progress_bar
import numpy as np
import h5py
from utils import *
from models.utils import get_model, get_criterion
from passers import Passer
from savers import save_activations, save_checkpoint, save_losses
from loaders import *
from graph import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--n_train_batches', default=0, type=int)
parser.add_argument('--input_size', default=32, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
''' Meta-name to be used as prefix on all savings'''
oname = args.net + '_' + args.dataset
save_path = '/data/data1/datasets/cvpr2019/adjacency/'+oname+'/'

''' If save directory doesn't exist create '''
if not os.path.exists(save_path):
    os.makedirs(save_path)    

# Build models
print('==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Prepare criterion '''
if args.dataset in ['cifar10', 'imagenet']:
    criterion = nn.CrossEntropyLoss()
elif args.dataset in ['mnist', 'mnist_adverarial']:
    criterion = F.nll_loss


for epoch in args.epochs:
    print('==> Loading checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    
    ''' Define passer and get activations'''
    functloader = loader(args.dataset+'_test', subset=list(range(0, 1000)))
    passer = Passer(net, functloader, criterion, device)
    activs = passer.get_function()

    for a in activs:
        adj = adjacency_correlation(np.transpose(np.reshape(a, (a.shape[0], -1))))
        np.savetxt(save_path+'badj_epc{}_trl{}.csv'.format(epoch, args.trial), adj, fmt='%.2f', delimiter=",")
