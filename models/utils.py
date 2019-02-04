from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *


def get_model(name, dataset):
    if name=='lenet' and dataset in ['mnist', 'cifar10', 'mnist_adversarial']:
        net = LeNet(num_classes=10)
    if name=='lenet' and dataset == 'imagenet':
        net = LeNet(num_classes=200)
    if name=='lenet32' and dataset in ['mnist', 'cifar10', 'mnist_adversarial']:
        net = LeNet(num_classes=10, input_size=32)
    if name=='lenet32' and dataset == 'imagenet':
        net = LeNet(num_classes=200, input_size=32)
    if name=='lenetext' and dataset=='mnist':
        net = LeNetExt(n_channels=1, num_classes=10)
    if name=='lenetext' and dataset=='cifar10':
        net = LeNetExt(n_channels=3, num_classes=10)
    if name=='vgg' and dataset in ['cifar10', 'mnist', 'vgg_cifar10_adversarial']:
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


def get_criterion(dataset):
    ''' Prepare criterion '''
    if dataset in ['cifar10', 'imagenet']:
        criterion = nn.CrossEntropyLoss()
    elif dataset in ['mnist', 'mnist_adverarial']:
        criterion = F.nll_loss
        
    return criterion 


def init_from_checkpoint(net):
    ''' Initialize from checkpoint'''
    print('==> Initializing  from fixed checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.net + '_' +args.dataset + '/ckpt_trial_' + str(args.fixed_init) + '_epoch_50.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_accc, start_epoch
