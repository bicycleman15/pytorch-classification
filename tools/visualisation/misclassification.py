import matplotlib.pyplot as plt
import seaborn as sn

import torchvision.datasets as datasets
dataloader = datasets.CIFAR10
testset = dataloader(root='./data', train=False, download=False, transform=None)


print(testset[2][0].save("testing.jpg"))

class plotter():
    def __init__(self, prefix):
        self.prefix = prefix