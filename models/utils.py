from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *



num_classes={'mnist':10,
         'cifar10':10,
         'cifar10_gray':10, 
         'mnist_adversarial':10,
         'imagenet':200,
         'imagenet_gray':200,
         'vgg_cifar10_adversarial':10}

'''
def get_model(name, dataset):
    if name=='lenet':
        net = LeNet(num_classes[dataset])
    if name=='lenet32':
        net = LeNet(num_classes[name])
    if name=='lenetext':
        net = LeNetExt()
'''

def get_model(name, dataset):
    if name=='lenet_300_100' and dataset in ['mnist', 'cifar10']:
        net = LeNet_300_100(num_classes=10)
    if name=='lenet' and dataset in ['mnist', 'cifar10', 'mnist_adversarial']:
        net = LeNet(num_classes=10)
    if name=='lenet' and dataset == 'imagenet':
        net = LeNet(num_classes=200)
    if name=='lenetbin':
        net = LeNet(num_classes=2)
    if name=='lenet32bin':
        net = LeNet(num_classes=2, input_size=32)
    if name=='lenet32' and dataset in ['mnist', 'cifar10_gray', 'mnist_adversarial']:
        net = LeNet(num_classes=10, input_size=32)
    if name=='lenet32bin':
        net = LeNet(num_classes=2, input_size=32)
    if name=='lenet32' and dataset == 'imagenet_gray':
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
    if dataset in ['cifar10', 'cifar10_gray', 'imagenet']:
        criterion = nn.CrossEntropyLoss()
    elif dataset in ['mnist', 'mnist_adversarial']:
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
