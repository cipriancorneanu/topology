from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *


def build_model(name, dataset):
    if name=='lenet' and dataset in ['mnist', 'cifar10', 'mnist_adversarial']:
        net = LeNet(num_classes=10)
    if name=='lenet' and dataset == 'imagenet':
        net = LeNet(num_classes=200)
    if name=='lenetext' and dataset=='mnist':
        net = LeNetExt(n_channels=1, num_classes=10)
    if name=='lenetext' and dataset=='cifar10':
        net = LeNetExt(n_channels=3, num_classes=10)
    if name=='vgg' and dataset in ['cifar10', 'mnist']:
        net = VGG('VGG16', num_classes=10)
    if name=='vgg' and dataset=='imagenet':
        net = VGG('VGG16', num_classes=200)
    if name=='resnet' and dataset in ['cifar10', 'mnist']:
        net = ResNet18(num_classes=10)
    if name=='resnet' and dataset=='imagenet':
        net = ResNet18(num_classes=200)
    if name=='densenet' and dataset=='cifar10':
        net = DenseNet121(num_classes=10)    
    if name=='densenet' and dataset=='imagenet':
        net = DenseNet121(num_classes=200)
    if name=='inception' and dataset=='cifar10':
        net = GoogLeNet(num_classes=10)
    if name=='inception' and dataset=='imagenet':
        net = GoogLeNet(num_classes=200)
    if name=='alexnet' and dataset=='cifar10':
        net = AlexNet(num_classes=10)
    if name=='alexnet' and dataset=='imagenet':
        net = AlexNet(num_classes=200)

    return net
