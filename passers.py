from utils import progress_bar
import torch
import numpy as np

''' Do subsampling and label shuffling in the data loader'''
'''if n_batches and batch_idx > n_batches: break'''
'''
if shuffle_labels and list(targets.size())[0]==128:
targets = targets[PERM]
'''

def get_accuracy(predictions, targets):
    ''' Compute accuracy of predictions to targets. max(predictions) is best'''
    _, predicted = predictions.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()

    return 100.*correct/total

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

class Passer():
    def __init__(self, net, loader, criterion, device):
        self.network = net
        self.criterion = criterion
        self.device = device
        self.loader = loader

    def _pass(self, optimizer=None, permute_labels=0):
        ''' Main data passing routing '''
        losses, features, total, correct = [], [], 0, 0
        accuracies = []
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            if permute_labels and list(targets.size())[0]==128:    
                a = int(permute_labels*list(targets.size())[0])                
                targets[:a] = targets[PERM[:a]]

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if optimizer: optimizer.zero_grad()
            outputs = self.network(inputs)

            loss = self.criterion(outputs, targets)
            losses.append(loss.item())
            
            if optimizer:
                loss.backward()
                optimizer.step()

            accuracies.append(get_accuracy(outputs, targets))
            
            progress_bar(batch_idx, len(self.loader), 'Mean Loss: %.3f | Last Loss: %.3f | Acc: %.3f%%'
                     % (np.mean(losses), losses[-1], np.mean(accuracies)))

        return np.asarray(losses), np.mean(accuracies)
    
    
    def run(self, optimizer=None, permute_labels=0):
        if optimizer:
            self.network.train()
            return self._pass(optimizer, permute_labels)
        else:
            self.network.eval()
            with torch.no_grad():
                return self._pass(permute_labels=permute_labels)

            
    def get_function(self):
        ''' Collect function (features) from the self.network.modeule.forward_features() routine '''
        features = []
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)
            features.append([f.cpu().data.numpy().astype(np.float16) for f in self.network.module.forward_features(inputs)])

            progress_bar(batch_idx, len(self.loader))

        return [np.concatenate(list(zip(*features))[i]) for i in range(len(features[0]))]

    def get_structure(self):
        ''' Collect structure (weights) from the self.network.module.forward_weights() routine '''
        weights = []
        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)
            weights.append([f.cpu().data.numpy().astype(np.float16) for f in self.network.module.forward_weights(inputs)])

            progress_bar(batch_idx, len(self.loader))
            
        return [np.concatenate(list(zip(*weights))[i]) for i in range(len(weights[0]))]


'''

def train(net, trainloader, device, optimizer, criterion, do_optimization, shuffle_labels, n_batches):
    net.train()
    train_loss, correct, total = 0, 0, 0
    loss_acc = []
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if n_batches and batch_idx > n_batches: break
            
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
'''
