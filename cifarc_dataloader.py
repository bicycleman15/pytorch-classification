from numpy.core.defchararray import index
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

class CIFAR100_C(data.Dataset):
    def __init__(self, path="data/CIFAR-100-C/fog.npy", severity=1, transform = None, target_transform = None):
        self.data = np.load(open(path, "rb"))
        self.label = np.load(open("data/CIFAR-100-C/labels.npy", "rb"))

        print("[INFO] loaded data {}".format(path))

        self.offset = (severity-1)*10000
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, idx):
        img, target = self.data[idx + self.offset], self.label[idx]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = torch.tensor(target).long()

        return img, target

    def __len__(self):
        assert(self.data.shape[0]//5 == 10000)
        return 10000

if __name__ == "__main__":
    dataset = CIFAR100_C()

    img, label = dataset[0]

    print(img)
    print(label)
    
