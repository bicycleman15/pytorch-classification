from .cifar.resnet import resnet34, resnet18, resnet50
from .cifar.alexnet import alexnet

model_dict = {
    "resnet34" : resnet34,
    "resnet50" : resnet50,
    "resnet18" : resnet18,
    "alexnet" : alexnet
}