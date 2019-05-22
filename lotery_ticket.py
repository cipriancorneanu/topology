from __future__ import print_function
import argparse
import pickle as pkl
import numpy as np
import h5py
from utils import *
from passers import Passer
from savers import save_activations, save_checkpoint, save_losses
from loaders import *
from labels import *
from prune import *
from models.utils import get_model, get_criterion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=20, help='resume from epoch')
parser.add_argument('--save_every', default=1, type=int)
parser.add_argument('--permute_labels', default=0, type=float)
parser.add_argument('--binarize_labels', default=-1, type=int)
parser.add_argument('--fixed_init', default=0, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--subset', default=0, type=float)
parser.add_argument('--random_importance', default=0, type=int)
parser.add_argument('--epochs_retrain', default=0, type=int)
parser.add_argument('--structured', default=0, type=int)
parser.add_argument('--perc', default=0.5, type=float)

args = parser.parse_args()

resume_epoch, eps = get_epc_eps(args.net, args.dataset, args.trial)

SAVE_EPOCHS = list(range(11)) + list(range(10, args.epochs+1, args.save_every)) # At what epochs to save train/test stats
ONAME = args.net + '_' + args.dataset # Meta-name to be used as prefix on all savings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

''' Prepare loaders '''
trainloader = loader(args.dataset+'_train', batch_size=args.train_batch_size, sampling=args.binarize_labels)
n_samples = len(trainloader)*args.train_batch_size
subset = list(np.random.choice(n_samples, int(args.subset*n_samples)))
subsettrainloader = loader(args.dataset+'_train', batch_size=args.train_batch_size, subset=subset, sampling=args.binarize_labels)
testloader = loader(args.dataset+'_test', batch_size=args.test_batch_size, sampling=args.binarize_labels)
criterion  = get_criterion(args.dataset)


''' Build models '''
print('==> Building model..')
net = get_model(args.net, args.dataset)
print(net)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Load checkpoint '''
print('==> Loading checkpoint for epoch {}...'.format(resume_epoch))
print('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(resume_epoch)+'.t7')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(resume_epoch)+'.t7')
net.load_state_dict(checkpoint['net'])

''' Optimization '''
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Define passer '''
if not args.subset:
    passer_train = Passer(net, trainloader, criterion, device)
else:
    passer_train = Passer(net, subsettrainloader, criterion, device)
    
passer_test = Passer(net, testloader, criterion, device)

'''passer_test.run()'''

''' Define manipulator '''
manipulator = load_manipulator(args.permute_labels, args.binarize_labels)

''' Get node importance '''
fname = 'node_importance_{}_{}.pkl'.format(args.net, args.dataset)
if os.path.isfile(fname):
    print('Load node importance from disk')
    with open(fname, 'rb') as handle:
        x = pkl.load(handle)

    node_importance = x['node_importance']
    importance = x['importance']
else:
    print('Compute node importance and save to disk for future use')
    functloader = loader(args.dataset+'_test', batch_size=100, subset = list(range(0,1000)))
    passer_funct = Passer(net, functloader, criterion, device)

    node_importance, importance = compute_node_importance_adj(net.module, passer_funct, eps, passer_test.get_sample())

    with open(fname, 'wb') as handle:
        pkl.dump({'node_importance': node_importance, 'importance': importance}, handle, protocol=pkl.HIGHEST_PROTOCOL)
     

''' Create mask for model according to topology '''
imp = np.zeros_like(node_importance)
for x,y in zip(node_importance, importance):
    imp[x] = y

nodes  = [x for x, _ in  evaluate_group_node_importance(net.module, passer_test.get_sample(), imp, random=args.random_importance)]
importance = [y for _, y in  evaluate_group_node_importance(net.module, passer_test.get_sample(), imp, random=args.random_importance)]
        
nodes_to_mask = [item for sublist in nodes[:int(len(nodes)*args.perc)] for item in sublist]
perc = 100*len(nodes_to_mask)/len(node_importance)
print('Print {0:2f}% nodes'.format(perc))

''' Apply masking on nodes '''        
mask = get_mask(net.module, passer_test.get_sample(),  nodes_to_mask)
    
loss_te, acc_te = passer_test.run()

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

net.apply(weights_init)
print('----Reinitialize---')
loss_te, acc_te = passer_test.run()


losses = []
for epoch in range(start_epoch, start_epoch+args.epochs):
    print('Epoch {}'.format(epoch))

    loss_tr, acc_tr = passer_train.run(optimizer, manipulator=manipulator, mask=mask)
    loss_te, acc_te = passer_test.run(mask=mask)
   
    losses.append({'loss_tr':loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te':acc_te})
    lr_scheduler.step(acc_te)

    if epoch in SAVE_EPOCHS:
        save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': epoch}, path='./checkpoint/'+ONAME+'/', fname='ckpt_trial_'+str(args.trial)+'_epoch_'+str(epoch)+'lt_'+str(perc)+'.t7')

save_losses(losses, path='./losses/'+ONAME+'/', fname='stats_trial_' +str(args.trial) +'lt_'+str(perc)+'.pkl')
