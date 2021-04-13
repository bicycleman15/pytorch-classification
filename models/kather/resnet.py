import torch.nn as nn
import torchvision.models as models
import torch

class ResNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet, self).__init__()

        self.pretrained_model = models.resnet50(pretrained= True)
        self.pretrained_model = torch.nn.Sequential(*(list(self.pretrained_model.children())[:-2]))

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        self.extra_conv = nn.Sequential(
            nn.Conv2d(in_channels = 2048, out_channels = 2048, kernel_size = 1),
            nn.BatchNorm2d(num_features = 2048),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 1)

            
        )
        
        self.fc1 = nn.Linear(in_features = 2048*6*6, out_features= 1024)
        self.drop_out = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features = 1024, out_features= num_classes)

    def forward(self, x):
        x = self.pretrained_model(x) # [batch, 2048, 7, 7]
        x = self.extra_conv(x) # [batch, 2048, 6, 6]
        x = x.view(x.size(0), -1)

        x = self.drop_out(x)
        x = self.relu(self.fc1(x)) # [batch, 1024]

        x = self.drop_out(x)
        x = self.relu(self.fc2(x)) # [batch, num_classes]
        
        return x