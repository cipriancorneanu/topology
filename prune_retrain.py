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
parser.add_argument('--prune_step', default=1, type=int)

args = parser.parse_args()

resume_epoch, eps = get_epc_eps(args.net, args.dataset, args.trial)
'''node_importance, importance = compute_node_importance(args.net, args.dataset, args.trial)'''



'''
node_importance = [396, 344, 320, 386, 397, 321, 300, 385, 149, 304, 257, 390, 365, 225, 325, 183, 335, 322,
  336, 308, 318, 273, 338,  82, 330, 367, 253, 102, 339, 306,  80, 307, 319, 314, 359, 295,
  377,  98, 401, 264, 170,  12, 333, 317, 243, 380, 326, 158, 364,   7, 366,  79, 327, 185,
  345, 216, 332, 383, 303, 376,  31,  32, 369, 346, 302, 298, 407, 241, 315,  43, 352, 275,
  342, 285, 214, 387,  73, 126, 363,  72, 312, 113, 310,  99, 305, 409,  44,  15, 361, 343,
   39,  58, 179, 394,  14, 131, 276, 283, 301, 199, 316, 388, 372, 171, 223, 331, 391, 274,
   97,  87, 290,  95, 340, 270,  27, 261,  81, 371, 395, 329, 162, 357,  63, 213,   2, 181,
  262,  75, 174, 244, 186, 101, 120, 229, 389, 211, 347, 163, 108, 210, 191, 196, 125,  68,
  349, 354, 313,  77, 280, 381, 278, 190, 351, 355,  46, 234,  83,  84, 138, 370,  47, 107,
  121,  28, 238, 219, 167,  89, 406, 378,   8,  96,  70, 360, 400, 263, 404, 382, 148, 119,
  159, 324, 334, 236, 117, 156, 288, 392, 142, 200,  16, 188,  22, 114,  34, 116, 165,  50,
  177,  21, 115,  78,  76,  13, 204, 226, 356, 205, 231, 279, 249, 221, 237, 182, 220, 180,
  271, 178,  17, 311,  30,  20, 398, 373, 215,  35, 137, 294, 111, 193, 309,  40,  42, 139,
  206,   1, 348,  11, 106, 296, 260,  33, 151, 155, 289,  74, 293, 202, 128,  55,   5,   6,
  192,  92, 402,   9,  94, 286, 269, 287, 132, 252,  60,  64,  62,  59,  65,  61, 408,  66,
   67,  69, 299,  71, 297,  57, 292, 291, 284, 282, 281,  85, 323,  41,  56, 379,  25,  24,
   23, 374, 375,  19,  18, 384,  54, 393, 399,  10, 403,   4,   3, 405,  26, 368,  29, 362,
   53,  52,  51, 328,  49,  48, 337, 341,  45, 350, 353,  38,  37,  36, 358,  86, 265,  88,
  233, 168, 228, 166, 230, 164, 232, 161, 160, 235, 227, 157, 154, 153, 152, 150, 239, 240,
  147, 169, 172, 145, 189, 201, 198, 197, 207, 195, 194, 208, 209, 212, 173, 187, 217, 184,
  218, 222, 176, 175, 224, 146, 144, 277, 103, 256, 258, 112, 110, 109, 259, 105, 104, 203,
  254, 266, 100, 267, 268, 272,  93,  91,  90, 255, 118, 143, 246, 242, 141, 140, 245, 136,
  135, 134, 133, 130, 251, 129, 127, 247, 248, 124, 123, 122, 250,   0]
'''

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
     
    
if args.structured:
    imp = np.zeros_like(node_importance)
    for x,y in zip(node_importance, importance):
        imp[x] = y

    stats = []
    cum_nodes = []
    nodes  = [x for x, _ in  evaluate_group_node_importance(net.module, passer_test.get_sample(), imp, random=args.random_importance)]
    importance = [y for _, y in  evaluate_group_node_importance(net.module, passer_test.get_sample(), imp, random=args.random_importance)]
        
    indices = list(range(0, len(nodes), args.prune_step))
    for indx_beg, indx_end in zip(indices[:-1], indices[1:]):
        nodes_slice = [item for sublist in nodes[indx_beg:indx_end] for item in sublist]
        cum_nodes.extend(nodes_slice)
        perc = 100*len(cum_nodes)/len(node_importance)
        print('Print {0:2f}% nodes'.format(perc))

        ''' Apply masking on nodes '''        
        mask = get_mask(net.module, passer_test.get_sample(),  cum_nodes)
              
        ''' Run test '''
        loss_te, acc_te = passer_test.run(mask=mask)
        
        ''' Retrain '''
        for epoch in range(args.epochs_retrain):
            loss_tr, acc_tr = passer_train.run(optimizer, manipulator=manipulator, mask=mask)
            loss_te, acc_te = passer_test.run(mask=mask)
            
        stats.append({'perc': perc, 'acc_te':acc_te})
        
    save_losses(stats, path='./losses/'+ONAME+'/', fname='prune_stats_trial_' +str(args.trial) + '_rand_' + str(args.random_importance) + '_retrain_' + str(args.epochs_retrain) + '_ps_' + str(args.prune_step) + 'structured.pkl')
else:
    stats = []
    for i in range(1, len(node_importance), 4):
        nodes = node_importance[-i:]
        perc = 100*len(nodes)/len(node_importance)
        print('Print {0:2f}% nodes'.format(perc))
    
        ''' Apply masking on nodes '''
        mask = get_mask(net.module, passer_test.get_sample(),  nodes)
        passer_test = Passer(net, testloader, criterion, device)
              
        ''' Run test '''
        loss_te, acc_te = passer_test.run(mask=mask)

        stats.append({'perc': perc, 'acc_te':acc_te})
    
        ''' Retrain '''
        for epoch in range(args.epochs_retrain):
            loss_tr, acc_tr = passer_train.run(optimizer, manipulator=manipulator)

    save_losses(stats, path='./losses/'+ONAME+'/', fname='prune_stats_trial_' +str(args.trial) + '_rand_' + str(args.random_importance) + '_retrain_' + str(args.epochs_retrain) + '_ps_' + str(args.prune_step) + '.pkl')

'''
losses = []
for epoch in range(start_epoch, start_epoch+args.epochs):
    print('Epoch {}'.format(epoch))

    loss_tr, acc_tr = passer_train.run(optimizer, manipulator=manipulator)
    loss_te, acc_te = passer_test.run()
   
    losses.append({'loss_tr':loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te':acc_te})
    lr_scheduler.step(acc_te)

    if epoch in SAVE_EPOCHS:
        save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': epoch}, path='./checkpoint/'+ONAME+'/', fname='ckpt_trial_'+str(args.trial)+'_epoch_'+str(epoch)+'.t7')

save_losses(losses, path='./losses/'+ONAME+'/', fname='stats_trial_' +str(args.trial) +'.pkl')
'''
