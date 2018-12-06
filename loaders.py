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


TRANSFORMS_TR_CIFAR10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TE_CIFAR10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TRANSFORMS_TR_IMAGENET = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))       
])

TRANSFORMS_TE_IMAGENET = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TRANSFORMS_MNIST_ADV = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

TRANSFORMS_MNIST = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])


def loader(data, subset=[]):
    ''' Interface to the dataloader function '''
    if data == 'mnist_train':
        return dataloader('mnist', './data', train=True, transform=TRANSFORMS_MNIST, batch_size=128, shuffle=False, num_workers=2, subset=subset)
    elif data == 'mnist_test':
        return dataloader('mnist', './data', train=False, transform=TRANSFORMS_MNIST, batch_size=100, shuffle=False, num_workers=2, subset=subset)
    elif data  == 'cifar10_train':
        return dataloader('cifar10', './data', train=True, transform=TRANSFORMS_TR_CIFAR10, batch_size=128, shuffle=False, num_workers=2, subset=subset)
    elif data == 'cifar10_test':
        return dataloader('cifar10', './data', train=False, transform=TRANSFORMS_TE_CIFAR10, batch_size=100, shuffle=False, num_workers=2, subset=subset)
    elif data == 'mnist_adversarial':
        return dataloader('/data/data1/datasets/lenet_mnist_adversarial/', train=False,
                          transform=TRANSFORMS_TE_CIFAR10, batch_size=100, shuffle=False, num_workers=2, subset=subset)
    elif data == 'cifar10_adversarial':
        return dataloader('/data/data1/datasets/lenet_cifar_adversarial/', train=False,
                          transform=TRANSFORMS_TE_CIFAR10, batch_size=100, shuffle=False, num_workers=2, subset=subset)
    elif data == 'imagenet_train':
        return dataloader('tinyimagenet', '/data/data1/datasets/tiny-imagenet-200/train/',
                                 train=True, transform=TRANSFORMS_TR_IMAGENET, batch_size=128, shuffle=True, num_workers=2, subset=subset)
    elif data == 'imagenet_test':
        return dataloader('tinyimagenet', '/data/data1/datasets/tiny-imagenet-200/val/images/',
                                 train=False, transform=TRANSFORMS_TE_IMAGENET, batch_size=100, shuffle=False, num_workers=2, subset=subset)


def dataloader(data, path, train, transform, batch_size, shuffle, num_workers, subset=[]):
    ''' Return loader for torchvision data '''
    if data == 'mnist':
        dataset = torchvision.datasets.MNIST(path, train=train, download=True, transform=transform)
    elif data == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(path, train=train, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    if subset : dataset = torch.utils.data.Subset(dataset, subset)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader





'''
def mnist_adversarial():
    dataset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/lenet_mnist_adversarial/', transform=TRANSFORM_MNIST_ADV)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    return loader


def cifar_adversarial():
    dataset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/cifar_adversarial/', transform=TRANSFORMS_TE_CIFAR10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    return loader
    

def mnist():
    print('===> Preparing data...')
    
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform_test)  
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def cifar10():
    print('===> Preparing data...')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def tinyimagenet(input_size=32):
    print('==> Preparing data...')

    trainset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/tiny-imagenet-200/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.ImageFolder(root='/data/data1/datasets/tiny-imagenet-200/val/images/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader
'''

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
