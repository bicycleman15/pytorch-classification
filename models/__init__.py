from .cifar.resnet import resnet32, resnet20
from .cifar.alexnet import alexnet

model_dict = {
    "resnet32" : resnet32,
    "alexnet" : alexnet,
    "resnet20": resnet20
}