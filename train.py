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
from models.utils import build_model
from utils import progress_bar
import numpy as np
import h5py
from utils import *
import pickle as pkl
from passers import Passer
from savers import save_activations, save_checkpoint, save_losses
from loaders import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=20, type=int, help='resume from epoch')
parser.add_argument('--save_every', default=10, type=int)
parser.add_argument('--shuffle_labels', default=0, type=int)
parser.add_argument('--fixed_init', default=0, type=int)
parser.add_argument('--n_train_batches', default=0, type=int)
parser.add_argument('--input_size', default=32, type=int)
args = parser.parse_args()


SAVE_EPOCHS = list(range(10)) + list(range(10, args.epochs, args.save_every)) # At what epochs to save train/test stats
ONAME = args.net + '_' + args.dataset # Meta-name to be used as prefix on all savings


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

''' Prepare loaders '''
trainloader = loader(args.dataset+'_train')
testloader = loader(args.dataset+'_test')
functloader = loader(args.dataset+'_test', subset=list(range(0, 1000)))

criterion  = get_criterion(args.dataset)
    
''' Build models '''
print('==> Building model..')
net = build_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Resume from checkpoint'''
if args.resume:
    net, best_acc, start_acc = init_from_checkpoint(net)
    
''' Optimization '''
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Define passer'''
passer_train = Passer(net, trainloader, criterion, device)
passer_test = Passer(net, testloader, criterion, device)
passer_functional = Passer(net, functloader, criterion, device)


''' Make intial pass before any training '''
activs, loss_te, acc_te = passer_test.run()
save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': 0}, path='./checkpoint/', fname='ckpt_trial_'+str(args.trial)+'_epoch_0.t7')
save_activations(activs, path='./activations/'+oname, fname='activations_trial_'+str(args.trial)+'.hdf5', internal_path='epoch_0')


losses = []
for epoch in range(start_epoch, start_epoch+args.epochs):
    print('Epoch {}'.format(epoch))
    loss_tr, acc_tr = passer_train.run(optimizer)
    '''loss_tr, acc_tr = train(net, trainloader, device, optimizer, criterion, do_optimization=False,  shuffle_labels=args.shuffle_labels, n_batches=args.n_train_batches)'''
    loss_te, acc_te = passer_test.run()
    activs = passer_functional.get_function()
    
    losses.append({'loss_tr':loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te':acc_te})
        
    lr_scheduler.step(acc_te)

    if epoch in SAVE_EPOCHS:
        save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': epoch}, path='./checkpoint/'+ONAME+'/', fname='ckpt_trial_'+str(args.trial)+'_epoch_'+str(epoch)+'.t7')
        save_activations(activs, path='./activations/'+ONAME+'/', fname='activations_trial_'+str(args.trial)+'.hdf5', internal_path='epoch_'+str(epoch))

'''Save losses'''
save_losses(losses, path='./losses/'+ONAME+'/', fname='stats_trial_' +str(args.trial) +'.pkl')
