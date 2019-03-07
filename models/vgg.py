'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import itertools

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    def forward_features(self, x):
        if self.name == 'VGG11': layers = [3, 7, 10, 14, 17, 21, 24, 27, 29]
        if self.name == 'VGG16': layers = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 44]
        return [self.features[:l+1](x) for l in layers] + [self.forward(x)]

    
    def forward_all_features(self, x):
        return [self.features[:l+1](x) for l in range(45)] + [self.forward(x)]

    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    
    def get_layers(self):
        layers = list(self.features.children())
        layers.append(self.classifier)
        return layers
        
        
def test():
    vgg11 = VGG('VGG11')
    vgg16 = VGG('VGG16')

    print(vgg16)
    x = torch.zeros([1,3,32,32])
    
    n_nodes = get_node_number(vgg16, x)
    print('Number of nodes in the network is {}'.format(n_nodes))

    ''' Build structure from model'''
    A = structure_from_view(vgg16, x, list(range(33, 44)))
    print(A.shape)
    
test()
