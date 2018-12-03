from utils import progress_bar


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
