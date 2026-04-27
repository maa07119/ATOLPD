import torch
from train import train_batches
import torchvision
from torchvision import transforms
import copy
import argparse
import numpy as np
import random
from models import mobilenet_v2
from models.resnet import ResNet18

# setting random seeds
seed = 12345
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', help = 'Model type (Currently avialable resnet18, mobilenet_v2)', default='resnet18')
    parser.add_argument('--dataset', help = 'name of dataset', default = 'cifar10')
    parser.add_argument('--device', help = 'GPU device to work on')
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--num_classes', type = int)
    parser.add_argument('--optimizer',  help = 'ADAM or SGD')
    parser.add_argument('--accuracy_threshold',  type = float)
    parser.add_argument('--pretrained_weights_path')

    args = parser.parse_args()

    dataset = args.dataset
    accuracy_threshold = args.accuracy_threshold
    num_classes = args.num_classes
    checkpoint_path = args.pretrained_weights_path
    

    dataset_image_size = {
        'cifar10': 32,
        'cifar100': 32,
        'SVHN': 32,
        'CINIC': 32
    }
    


    image_size = dataset_image_size[dataset]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
    ])
    
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=test_transform)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
                    root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(
                    root='./data', train=False, download=True, transform=test_transform)
    elif dataset == 'SVHN':
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=train_transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=test_transform)
    elif dataset == 'CINIC':
        cinic_directory = "data/CINIC-10"
        trainset = torchvision.datasets.ImageFolder(cinic_directory + '/train', transform=train_transform)
        testset = torchvision.datasets.ImageFolder(cinic_directory + '/test', transform=test_transform)

    print(dataset)
    # dataset_configs = batches_lr_sgd[dataset] 
    dataset_percentage = 0.25 # Consider real case scenario with 25% of dataset.
    batch_sizes = [4, 8, 16, 32, 64]
    b_128_lr = 0.0005
    # Sqrt scaling for learning rate
    batches_lrs = {b: b_128_lr * ((b / 128)**0.5) for b in batch_sizes } 
    train_split = dataset_percentage
    new_train_set, _  = torch.utils.data.random_split(trainset, [train_split, 1 - train_split])
    new_test_set, _  = torch.utils.data.random_split(testset, [train_split, 1 - train_split])
    

    print('Dataset Len:', len(new_train_set), len(new_test_set))
    print("Batches lrs:", batches_lrs)
    # Train multiple batches till accuracy threshold (convergence) for given task.
    
    r = train_batches(args.model_type, checkpoint_path, num_classes, batches_lrs, args.optimizer,
        new_train_set, new_test_set, args.epochs, args.device, accuracy_thresold=accuracy_threshold)

    print(f"Proxy Dataset r = {r}")

