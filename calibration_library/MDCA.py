import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# @neelabh17 implementation


class MDCA(torch.nn.Module):
    def __init__(self):
       
        super(MDCA,self).__init__()


    def reset(self):
        pass


    def forward(self , output, target):

        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image

        create an three array of [n_class, n_bins]
        -> Number of prediciton array for that specification
        -> Number of correct prediction for that class
        -> Percentge of correct 
        '''

        output = torch.softmax(output, dim=1)
        # [batch, classes]

        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape

        for c in range(classes):
            avg_acc = (target == c).float().mean()
            avd_conf = torch.mean(output[:,c])
            loss += torch.abs(avd_conf-avg_acc)

        loss /= classes

        return loss

class Focal_MDCA(torch.nn.Module):
    def __init__(self, gamma=1.0):
       
        super(Focal_MDCA,self).__init__()
        self.gamma = gamma


    def reset(self):
        pass


    def forward(self , output, target):

        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image

        create an three array of [n_class, n_bins]
        -> Number of prediciton array for that specification
        -> Number of correct prediction for that class
        -> Percentge of correct 
        '''

        output = torch.softmax(output, dim=1)
        # [batch, classes]

        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape

        for c in range(classes):
            avg_acc = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += ((1 - avg_conf)**self.gamma) * torch.abs(avg_conf-avg_acc)

            # if(c == 3 and abs(avg_acc.item() - avg_conf.item()) >= 0.1):
            #     print(avg_acc.item(), avg_conf.item())

        loss /= classes

        return loss
