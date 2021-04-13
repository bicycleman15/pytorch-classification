import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# @neelabh17 implementation


class DCA(torch.nn.Module):
    def __init__(self):
       
        super(DCA,self).__init__()


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

        conf, pred_labels = torch.max(output, dim = 1)

        return torch.abs(conf.mean() -  (pred_labels == target).float().mean())

