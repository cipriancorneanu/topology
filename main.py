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
from models.densenet import *
from utils import progress_bar
import numpy as np
import h5py
from utils import *
import pickle as pkl

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

if args.dataset=='cifar':
    trainloader, testloader = prepare_cifar10()
    criterion = nn.CrossEntropyLoss()
elif args.dataset=='imagenet':
    trainloader, testloader = prepare_tiny_imagenet(args.input_size)
    criterion = nn.CrossEntropyLoss()
elif args.dataset=='mnist':
    trainloader, testloader = prepare_mnist()
    criterion = F.nll_loss
elif args.dataset=='mnist_adversarial':
    testloader = prepare_mnist_adversarial()
    criterion = F.nll_loss
    
oname = args.net + '_' + args.dataset

# Model
print('==> Building model..')
if args.net=='lenet' and args.dataset in ['mnist', 'cifar', 'mnist_adversarial']:
    net = LeNet(num_classes=10)
if args.net=='lenetext' and args.dataset=='mnist':
    net = LeNetExt(n_channels=1, num_classes=10)
if args.net=='lenetext' and args.dataset=='cifar':
    net = LeNetExt(n_channels=3, num_classes=10)
elif args.net=='vgg' and args.dataset in ['cifar', 'mnist']:
    net = VGG('VGG16', num_classes=10)
elif args.net=='vgg' and args.dataset=='imagenet':
    net = VGG('VGG16', num_classes=200)
elif args.net=='resnet' and args.dataset in ['cifar', 'mnist']:
    net = ResNet18(num_classes=10)
elif args.net=='resnet' and args.dataset=='imagenet':
    net = ResNet18(num_classes=200)
elif args.net=='densenet' and args.dataset=='imagenet':
    net = DenseNet()
elif args.net=='alexnet' and args.dataset=='cifar':
    net = AlexNet(num_classes=10)
elif args.net=='alexnet' and args.dataset=='imagenet':
    net = AlexNet(num_classes=200)
    
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.fixed_init:
    # Load checkpoint.
    print('==> Initializing  from fixed checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.net + '_' +args.dataset + '/ckpt_trial_' + str(args.fixed_init) + '_epoch_50.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.net + '_' +args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(args.resume_epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
        
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Create dataset for dumping activations '''
file = h5py.File('./activations/'+oname+'/activations_trial_' + str(args.trial) + '.hdf5', 'a')

activs, targets, loss_te, acc_te = test(net, testloader, device, criterion, n_test_batches=10)
save_model(net, acc_te, oname, args.trial, epoch=0)
save_activations(file, 0, activs, targets)

stats = []
for epoch in range(start_epoch, start_epoch+args.epochs+1):
    print('Epoch {}'.format(epoch))
    _, _ = train(net, trainloader, device, optimizer, criterion, do_optimization=True, shuffle_labels=args.shuffle_labels, n_batches=args.n_train_batches)
    loss_tr, acc_tr = train(net, trainloader, device, optimizer, criterion, do_optimization=False,  shuffle_labels=args.shuffle_labels, n_batches=args.n_train_batches)
    activs, targets, loss_te, acc_te = test(net, testloader, device, criterion, n_test_batches=10)

    stats.append({'loss_tr':loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te':acc_te})
        
    lr_scheduler.step(acc_te)

    if epoch<10:
        save_model(net, acc_te, oname, args.trial, epoch)
        save_activations(file, epoch, activs, targets)  
    elif epoch%args.save_every==0:
        save_model(net, acc_te, oname, args.trial, epoch)
        save_activations(file, epoch, activs, targets)

'''Save losses'''
with open('./losses/' + oname + '/' + 'stats_trial_' +str(args.trial) +'.pkl', 'wb') as f:
    pkl.dump(stats, f, protocol=2)

file.close()

