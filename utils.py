'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision
import numpy as np
import h5py
import errno
import os.path
import random

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def prepare_mnist_adversarial():
    transform_test = transforms.Compose([transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    testset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/lenet_mnist_adversarial/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)  

    return testloader

def prepare_cifar_adversarial():

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/cifar_adversarial/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)  

    return test_loader
    

def prepare_mnist():
    print('===> Preparing data...')
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
    
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform_test)   
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

    
def prepare_cifar10():
    # Data
    print('===> Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def prepare_tiny_imagenet(input_size=32):
    print('==> Preparing data...')
    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))       
    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/tiny-imagenet-200/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/tiny-imagenet-200/val/images/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def prepare_validation_imagenet(data_dir, validation_labels_file):
    
    # Read the synsets  associated with the validation data set.
    labels = [l.strip().split('\t')[1] for l in open(validation_labels_file).readlines()]
    unique_labels = set(labels)

    # Make all sub-directories in the validation data dir.
    for label in unique_labels:
        labeled_data_dir = os.path.join(data_dir, label)
        
        # Catch error if sub-directory exists
        try:
            os.makedirs(labeled_data_dir)
        except OSError as e:
            # Raise all errors but 'EEXIST'
            if e.errno != errno.EEXIST:
                raise

    # Move all of the image to the appropriate sub-directory.
    for i in range(len(labels)):
        basename = 'val_%d.JPEG' % i
        original_filename = os.path.join(data_dir, basename)
        if not os.path.exists(original_filename):
            print('Failed to find: %s' % original_filename)
            sys.exit(-1)
        new_filename = os.path.join(data_dir, labels[i], basename)    
        os.rename(original_filename, new_filename)

PERM = [30,  88,  93,  10,  53,  28,  22,   9, 116,  20,  51, 106,  14,
                76, 108, 119,  67,  82,  27,  39, 110,  68,  91,  74,  86, 117,
                56,  99,  66,  49,  11,  61,  65,   7,  58,  31,  35,  47,  23,
                96,  77,  25, 109,  12,  71, 123,  95,  48,  17,  44,   2, 101,
               118,   5,  59,  32, 122,  83,  78,  55,  54, 121,   8, 125,  97,
                57, 105, 120,  26,  43,  72,  40,  19, 115,  94,  89,  81,  64,
                70,  87,  29,  42,  46,  60,  37, 113,  41,   0,  92, 100,  24,
                75,  52,  90, 103,  84,   1,  80,  21, 111,   6,   3,   4,  79,
                50, 102, 112, 104,  45,  18,  62,  33, 127,  16,  38,  63,  85,
               124,  98,  36, 107,  73,  69,  34,  15, 114, 126,  13]


def train(net, trainloader, device, optimizer, criterion, do_optimization, shuffle_labels, n_batches):
    net.train()
    train_loss, correct, total = 0, 0, 0
    loss_acc = []
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if n_batches and batch_idx>n_batches: break
            
        if shuffle_labels and list(targets.size())[0]==128:
            targets = targets[PERM]
                
        inputs, targets = inputs.to(device), targets.to(device)
                
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss_acc.append(loss.item())

        loss.backward()
        optimizer.step()
                
        train_loss += loss.item()
        _, predicted = outputs.max(1)
                
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.*correct/total
    
    '''print('Train accuracy {}'.format(100.*correct/total))'''
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return np.asarray(loss_acc), accuracy

def train_subset(net, trainloader, device, optimizer, criterion, do_optimization, shuffle_labels, n_batches):
    net.train()
    train_loss, correct, total = 0, 0, 0
    loss_acc = []
    if do_optimization:
        n_repeats = int((60000/128)/n_batches)
    else:
        n_repeats = 1

    
    for repeat in range(0, n_repeats):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if n_batches and batch_idx>n_batches: break
            
            if shuffle_labels and list(targets.size())[0]==128:
                targets = targets[PERM]
                
            inputs, targets = inputs.to(device), targets.to(device)
                
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss_acc.append(loss.item())

            if do_optimization:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
                
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.*correct/total
    
    '''print('Train accuracy {}'.format(100.*correct/total))'''
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return np.asarray(loss_acc), accuracy


def test(net, testloader, device, criterion, n_test_batches):
    net.eval()
    test_loss, correct, total, target_acc, activation_acc = 0, 0, 0, [], []
    loss_acc = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):            
            inputs, targets = inputs.to(device), targets.to(device)            
            outputs = net(inputs)

            if batch_idx < n_test_batches:
                activations = [a.cpu().data.numpy().astype(np.float16) for a in net.module.forward_features(inputs)]
                target_acc.append(targets.cpu().data.numpy())
                activation_acc.append(activations)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            loss_acc.append(loss.item())
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = 100.*correct/total
                 
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                     % (test_loss/(batch_idx+1), accuracy))
          
    activs = [np.concatenate(list(zip(*activation_acc))[i]) for i in range(len(activation_acc[0]))]
    
    return (activs, np.concatenate(target_acc), np.asarray(loss_acc), accuracy)


def save_model(net, acc, save_name, trial, epoch):
    print('Saving checkpoint...')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+save_name+'/ckpt_trial_'+str(trial)+'_epoch_'+str(epoch)+'.t7')

        
def save_activations(file, epoch, activs, targets):
    print('Saving activations...')

    for i, x in enumerate(activs):
        file.create_dataset("epoch_"+str(epoch)+"/activations/layer_"+str(i), data=x, dtype=np.float16)
    file.create_dataset("epoch_"+str(epoch)+"/targets", data=targets)
