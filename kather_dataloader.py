import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import torchvision.models as models
from models.kather.resnet import ResNet

class Kather(data.Dataset):
    def __init__(self, root="data/Kather_texture_2016_image_tiles_5000", split=None, transform = None, target_transform = None):
        if split == "train":
            _split_f = os.path.join(root, "train.npy")
        elif split == "val":
            _split_f = os.path.join(root, "val.npy")
        else:
            raise RuntimeError('Unknown dataset split.')
        files_list = np.load(_split_f)
        self.images_loc = np.vectorize(lambda x: os.path.join(root, x))(files_list)
        self.labels = np.vectorize(lambda x: int(x[:2]))(files_list)
        self.transform = transform



    def __getitem__(self, idx):
        img, target = Image.open(self.images_loc[idx]), self.labels[idx]
        
        # img, target = cv2.imread(self.images_loc[idx]), self.labels[idx]
        
        # because classes in indexing 1 to 8
        target = torch.tensor(target).long() - 1 


        # img, target = self.data[idx + self.offset], self.label[idx]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        

        return img, target

    def __len__(self):
        assert (len(self.images_loc) == len (self.labels)) 
        return len(self.labels)

    @property
    def classes(self):
        """Category names."""
        return ('TUMOR', 'STROMA', 'COMPLEX', 'LYMPHO', 'DEBRIS', 'MUCOSA', 'ADIPOSE', 'EMPTY')


if __name__ == "__main__":
    import torchvision.transforms as transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(size = 100),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.ToTensor(),
        transforms.Normalize((0.65010238, 0.47286776, 0.58460745), (0.25447648, 0.32640151, 0.26681215)),
    ])

    trainset = Kather(root="data/Kather_texture_2016_image_tiles_5000", split = "val", transform=transform_train)
    print(len(trainset))
    trainloader = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)

    model = ResNet(num_classes = 8).cuda()
    print(model)

    for name, param in model.named_parameters():
            print(name,param.requires_grad)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    

    
    # finding out mean
    # overall = np.array([]).reshape(0,3,224,224)
    for i, (imgs, targets) in enumerate(trainloader):
    #     overall= np.concatenate((overall, imgs.numpy()), axis =0)
        out = model(imgs.cuda())
        print(out.shape)
    # print(np.mean(overall, axis = (0,2,3)))
    # print(np.std( overall, axis = (0,2,3)))


    
