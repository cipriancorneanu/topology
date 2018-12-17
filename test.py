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
import pickle as pkl
from models.utils import build_model
from passers import Passer
from savers import save_activations, save_checkpoint, save_losses
from loaders import *
from graph import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--n_train_batches', default=0, type=int)
parser.add_argument('--input_size', default=32, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
''' Meta-name to be used as prefix on all savings'''
oname = args.net + '_' + args.dataset

# Build models
print('==> Building model..')
net = build_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
''' Load checkpoint'''
print('==> Loading checkpoint...')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+args.net + '_' +args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(args.epoch)+'.t7')
net.load_state_dict(checkpoint['net'])

''' Define loader and passer'''
loader = loader(args.dataset+'_test', subset=list(range(0, 1000)))
passer = Passer(net, loader, criterion, device)
activs = passer.get_function()

for a in activs:
    adj = adjacency_correlation(np.transpose(np.reshape(a,(a.shape[0],-1))))

    ''' TO ADD SAVING'''
    '''
    np.savetxt(save_path+'badj_epc{}_trl{}_{:.3f}.csv'.format(epc, trial, dens), badj, fmt='%d', delimiter=",")
    '''
