from .cifar10 import get_train_valid_loader as cifar10loader
from .svhn import get_train_valid_loader as svhnloader
from .cifar100 import get_train_valid_loader as cifar100loader

from .cifar10 import get_val_temp_loader as cifar10loadert
from .svhn import get_val_temp_loader as svhnloadert
from .cifar100 import get_val_temp_loader as cifar100loadert

from .cifar10 import get_datasets as cifar10_ds
from .svhn import get_datasets as svhn_ds
from .cifar100 import get_datasets as cifar100_ds

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

temploader_dict = {
    "cifar10" : cifar10loadert,
    "cifar100" : cifar100loadert,
    "svhn" : svhnloadert
}

dataset_dict = {
    "cifar10" : cifar10_ds,
    "cifar100" : cifar100_ds,
    "svhn" : svhn_ds
}

dataset_class_dict = {
    "cifar10" : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    "cifar100" : ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 
                'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm'] ,

    "svhn" : [f"{i}" for i in range(10)]
}