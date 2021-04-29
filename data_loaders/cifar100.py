import os
import torch
import random

from torchvision import datasets
from torchvision import transforms
from torch.utils import data

def get_train_valid_loader(args):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataloader = datasets.CIFAR100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    indices = list(range(len(trainset)))

    random.seed(1)
    random.shuffle(indices)

    train_indices = indices[ int(0.1*len(trainset)) : ]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch_size, num_workers=args.workers, sampler=train_sampler)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    return trainloader, testloader

    
def get_datasets(args):
    dataloader = datasets.CIFAR100
    trainset = dataloader(root='./data', train=True, download=True, transform=None)
    testset = dataloader(root='./data', train=False, download=False, transform=None)

    return trainset, testset

def get_val_temp_loader(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = datasets.CIFAR100

    tempset = dataset(root='./data', train=True, download=True, transform=transform_train)
    # print(tempset)

    indices = list(range(len(tempset)))

    random.seed(1)
    random.shuffle(indices)

    temp_indices = indices[:int(0.1 * len(tempset))]
    temp_sampler = torch.utils.data.sampler.SubsetRandomSampler(temp_indices)

    # keep shuffle false since we have a random sampler
    temploader = data.DataLoader(tempset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.workers, sampler=temp_sampler)

    valset = dataset(root='./data', train=False, download=False, transform=transform_test)
    valloader = data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    return temploader, valloader