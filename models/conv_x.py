import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Conv_2(nn.Module):
    def __init__(self, num_classes, input_size=28):
        super(Conv_2, self).__init__()
        self.feat_size = 12544 if input_size==32 else 9216 if input_size==28 else -1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(self.feat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x2 = x2.view(-1, self.feat_size)
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = F.log_softmax(self.fc3(x4), dim=1)
        return x5
    
    def forward_features(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))        
        x2 = x2.view(-1, torch.prod(torch.tensor(x2.size())))
        x3 = F.relu(self.fc1(x2))
        x4 = F.relu(self.fc2(x3))
        x5 = F.log_softmax(self.fc3(x4), dim=1)
        return [x1, x2, x3, x4, x5]

        
class Conv_4(nn.Module):
    def __init__(self, num_classes, input_size=28):
        super(Conv_4, self).__init__()
        self.feat_size = 3200 if input_size==32 else 2048 if input_size==28 else -1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(self.feat_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(F.max_pool2d(self.conv4(x3), 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        x6 = F.relu(self.fc2(x5))
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return x7
    
    def forward_features(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(F.max_pool2d(self.conv2(x1), 2))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(F.max_pool2d(self.conv4(x3), 2))
        x4 = x4.view(-1, self.feat_size)
        x5 = F.relu(self.fc1(x4))
        x6 = F.relu(self.fc2(x5))
        x7 = F.log_softmax(self.fc3(x6), dim=1)
        return [x1, x2, x3, x4, x5, x6, x7]


def test():
    net = Conv_4(num_classes=10, input_size=28)
    print(net)
    x = torch.randn(1,1,28,28)
    y = net(x)

    for i, layer in enumerate(net.forward_features(x)):
        print('layer {} has size {}'.format(i, layer.shape))

test()
