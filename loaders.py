'''
Data loading Utilities for preparing for various different datasets.
Includes, MNIST, CIFAR10, TinyImagenet.
For MNIST and CIFAR10 there are special adversarial samples prepared
for evaluation. -> <dataset>_adversarial(). Each function returns a
train and a test DataLoader except the dedicated functions for adversarial
samples that return a single loader.
'''

import torchvision.transforms as transforms
import torch
import torchvision


def mnist_adversarial():
    ''' Return loader for adversarial mnist '''
    transform = transforms.Compose([transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    dataset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/lenet_mnist_adversarial/', transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    return loader


def cifar_adversarial():
    ''' Return loader for adversarial CIFAR10'''
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/cifar_adversarial/', transform=transform_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    return loader
    

def mnist():
    ''' Return loaders for MNIST '''
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

    
def cifar10():
    ''' Return loaders for CIFAR10'''
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

    ''' Not used '''
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def tinyimagenet(input_size=32):
    ''' Return loaders (train, test) for TinyImagenet'''
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


def validation_imagenet(data_dir, validation_labels_file):
    ''' What was this for? Is it used? '''

    import errno
    import os
    import sys
    
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
