import os
import torch
import numpy as np
import random

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data

def get_train_valid_loader(args):
    
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    data_dir = './data'
    train_dataset = datasets.SVHN(
        root=data_dir, split='train',
        download=True, transform=valid_transform,
    )

    valid_dataset = datasets.SVHN(
        root=data_dir, split='test',
        download=True, transform=valid_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        num_workers=args.workers, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.test_batch_size,
        num_workers=args.workers, shuffle=False
    )

    return (train_loader, valid_loader)


def get_datasets(args):
    
    # load the dataset
    data_dir = './data'
    train_dataset = datasets.SVHN(
        root=data_dir, split='train',
        download=True, transform=None,
    )
    valid_dataset = datasets.SVHN(
        root=data_dir, split='test',
        download=True, transform=None,
    )
    return (train_dataset, valid_dataset)

def get_val_temp_loader(args):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = './data'
    tempset = datasets.SVHN(
        root=data_dir, split='train',
        download=True, transform=valid_transform,
    )

    indices = list(range(len(tempset)))
    random.shuffle(indices)

    temp_indices = indices[:int(0.1 * len(tempset))]
    temp_sampler = torch.utils.data.sampler.SubsetRandomSampler(temp_indices)

    temploader = data.DataLoader(tempset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.workers, sampler=temp_sampler)

    valset = datasets.SVHN(
        root=data_dir, split='test',
        download=True, transform=valid_transform,
    )
    valloader = data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)

    return temploader, valloader