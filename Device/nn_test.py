import torch
import csv
import time
from threading import Event
import os
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from datasets.shakespeare import ShakespeareDataset
from models import resnet, mobilenet_v2, densenet, lstm, transformer
from power_monitor import PowerMonitor
import config
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_network_power_stats(device_name, static_cpu_frequency, static_gpu_frequency,
                            mode='TRAIN', model_type='resnet18', ml_device='GPU',
                            batch_size=32, epochs=1, m = 100):
    print("HERE")
    # get device read info
    device_info = config.devices_channels[device_name]
    # initialize power monitor thread and control event
    monitor_event = Event()
    power_monitor = PowerMonitor(event=monitor_event,
                                 device_config=device_info,
                                 static_cpu_frequency = static_cpu_frequency,
                                 static_gpu_frequency = static_gpu_frequency)

    # initialize model
    if model_type == 'resnet18':
        model = resnet.ResNet18()
    elif model_type == 'resnet50':
        model = resnet.ResNet50()
    elif model_type == 'mobilenet_v2':
        model = mobilenet_v2.MobileNetV2()
    elif model_type == 'densenet':
        model = densenet.densenet_cifar()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])
    
    if mode == 'TRAIN':
        # download dataset and set monitor sleeping time to 1 sec
        if model_type in config.vision_models:
            print('Loading cifar 10 dataset')
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=train_transform)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=0)
            print('Download test set')
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
            print('Done')
        elif model_type in config.language_models:
            print('Loading shakespeare dataset')
            trainset = ShakespeareDataset('shakespeare.txt')
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)

        power_monitor.sleep_time = 1
        print('Number of mini batches:', len(trainloader))

    # start monitoring Power
    print('Power monitor started')
    power_monitor.start()

    if ml_device == 'GPU':
        model = model.cuda()

    
    print(f'{mode} started ...')
    batch_time = train(model, trainloader, ml_device, epochs, power_monitor, testloader, m)
    # stop monitoring
    monitor_event.set()
    power_monitor.join()
    peak_power = power_monitor.get_peak_power()

    return batch_time, peak_power


def train(model, trainloader, device, epochs, power_monitor, testloader, m):
    warmup_batches = 5
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    for epoch in range(epochs):
        trainloader_tqdm = tqdm(trainloader)
        for batch_idx, (inputs, targets) in enumerate(trainloader_tqdm):
            if batch_idx == warmup_batches:
                start_time = time.time() # start recording time

            if device == 'GPU':
                inputs, targets = inputs.cuda(), targets.cuda()
                #print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            trainloader_tqdm.set_description(f"Epoch [{epoch}/{epochs}]")
            trainloader_tqdm.set_postfix(loss = loss.item())
            if batch_idx == m + warmup_batches:
               break
        #acc = test(model, testloader, device, power_monitor) # Comments as we are profiling training only 
    print('Training ended')
    end_time = time.time() - start_time
    batch_time = end_time / m 
    return batch_time
 
def test(model, testloader, device, power_monitor):
    print('TESTING...')
    print(len(testloader))

    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_losses = []
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if device == 'GPU':
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_losses.append(loss.item())
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        print(correct, len(testloader.dataset))
        test_loss = np.array(test_losses).mean()
        acc = correct / (len(testloader.dataset))
        print(f'test loss: {test_loss}, acc: {acc} ')
    return acc
