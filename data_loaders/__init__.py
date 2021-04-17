from .cifar10 import get_train_valid_loader as cifar10loader
from .svhn import get_train_valid_loader as svhnloader
from .cifar100 import get_train_valid_loader as cifar100loader

dataloader_dict = {
    "cifar10" : cifar10loader,
    "cifar100" : cifar100loader,
    "svhn" : svhnloader
}

dataset_nclasses_dict = {
    "cifar10" : 10,
    "cifar100" : 100,
    "svhn" : 10
}