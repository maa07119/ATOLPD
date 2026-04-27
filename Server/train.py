import torch
import torchvision
from torchvision import transforms
import numpy as np
import random
from models import resnet_cifars, mobilenet_v2
from models.resnet import ResNet18
from tqdm import tqdm


def initialize_model(model_type, num_classes, pretrained = True, checkpoint_path = None, orig_classes = 100):
    if model_type == 'resnet18':
        # Load pretrained weights
        if pretrained:
            model = ResNet18(orig_classes)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.linear = torch.nn.Linear(model.linear.in_features, num_classes, bias = True)
            print("pretrained model loaded")
        else:
        # From scratch
            model = ResNet18(num_classes)
    elif model_type == 'mobilenet_v2':
        # Load pretrained weights
        if pretrained:
            model = mobilenet_v2.MobileNetV2(orig_classes)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.linear = torch.nn.Linear(model.linear.in_features, num_classes, bias = True)
            print("pretrained model loaded")
        else:
        # From scratch
            model = mobilenet_v2.MobileNetV2(num_classes)
    return model

def train_batches(model_type, checkpoint_path, num_classes, batches_lr, optimizer_type, train_set, test_set, epochs, device, accuracy_thresold):

    N = 3 # Number of trails 
    r = np.zeros((len(batches_lr), N))
    # Train for N times
    for idx in range(N):
        for b_idx, (batch_size, lr) in enumerate(batches_lr.items()):
            model = initialize_model(model_type, num_classes, checkpoint_path=checkpoint_path)
            model = model.to(device)
            if optimizer_type == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)
            elif optimizer_type == 'ADAM':
                optimizer = torch.optim.Adam(model.parameters(), lr = lr)
            N_s = train_model_batch(
                            model, optimizer,
                            batch_size, train_set, test_set,
                            epochs = epochs, eval_mode = 'steps', device = device,
                            accuracy_thresold = accuracy_thresold,
                            scheduler = None)
            r[b_idx, idx] = N_s
    r = np.mean(r, axis= 1)
    r = r / np.max(r)
    return r 
            

def train_model_batch(model, optimizer, batch_size, train_set, test_set, writer = None, epochs = 30,
                    eval_mode = 'epochs', device = 'cuda:0', accuracy_thresold = None, scheduler = None):

    # Train loader with batch size
    trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    # Train till accuracy threshold
    N_s = train_steps(model, optimizer, trainloader, epochs, testloader, batch_size,
        iter_sample = 4096,
        accuracy_thresold = accuracy_thresold, device = device, scheduler = scheduler)
    return N_s


def train_steps(model, optimizer, trainloader, epochs,
                testloader, batch_size, iter_sample = 4096, accuracy_thresold = None,
                device = 'cuda:0', scheduler = None):

    criterion = torch.nn.CrossEntropyLoss()
    iterations = 0
    steps_loss = 0

    model.train()
    for epoch in range(epochs):
        trainloader_tqdm = tqdm(trainloader)
        for batch_idx, (inputs, targets) in enumerate(trainloader_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            trainloader_tqdm.set_description(f"Epoch [{epoch}/{epochs}]")
            trainloader_tqdm.set_postfix(loss = loss.item())
            steps_loss += loss.item()
            iterations += batch_size
            if iterations % iter_sample == 0:
                steps_loss /= (iter_sample / batch_size)
                print('Steps loss:', steps_loss)
                acc, test_loss = evaluate(model, testloader, epoch, device = device)


                # return the N_s (number of samples proccessed / iter_sample) till accuracy
                if accuracy_thresold is not None and acc >= accuracy_thresold:
                    print("Accuracy reached")
                    return iterations / iter_sample
                
                # clear loss and enable training after evaluation
                steps_loss = 0
                model.train()
        if scheduler:
            scheduler.step()
        
    return iterations / iter_sample

def evaluate(model, testloader, epoch, device = 'cuda:0'):
    model.eval()
    running_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        testloader_tqdm = tqdm(testloader)
        for batch_idx, (inputs, targets) in enumerate(testloader_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        test_loss =  running_loss/len(testloader)
        
        acc = correct / len(testloader.dataset)
        print(f'epoch: {epoch}, acc: {acc}, test loss: {test_loss}')
    return acc, test_loss





def filter_classes(dataset, no_classes, start_class, end_class):
    
    # classes_idx = np.where(np.array(dataset.targets) < no_classes)[0]
    targets_np = np.array(dataset.targets)
    # print(np.logical_and((start_class < targets_np), (targets_np < end_class)))
    classes_idx = np.where(np.logical_and((start_class <= targets_np), (targets_np < end_class)))[0]
    # print(classes_idx)
    # convert selected classes to 0-25
    for i in range(len(dataset.targets)):
        if i in classes_idx:
            dataset.targets[i] -= start_class
    
    subset = torch.utils.data.Subset(dataset, classes_idx)
    return subset
