import os
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(batch_size,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the SVHN dataset. 
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    
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
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)