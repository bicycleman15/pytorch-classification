import torch
import torch.nn as nn
from calibration_library.cce_loss import CCELossFast
from calibration_library.ece_loss import ECELoss
from calibration_library.MDCA import MDCA, Focal_MDCA
from calibration_library.DCA import DCA

class CCETrainLoss(CCELossFast):
    def __init__(self, n_classes, n_bins = 10, **kwargs):
        super().__init__(n_classes, n_bins=n_bins, mode = "train")

    def forward(self, output, target):
        # import pdb; pdb.set_trace()
        loss = super().forward(output, target)

        return dict(loss=loss)

class CCETrainLoss_alpha(CCELossFast):
    def __init__(self, n_classes, alpha, n_bins = 10, ignore_index=-1,**kwargs):
        super().__init__(n_classes, n_bins=n_bins, mode = "train")
        self.nll = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.alpha = alpha
        print("Using alpha value = {}".format(self.alpha))

    def forward(self, output, target):
        # import pdb; pdb.set_trace()
        loss_cal = super().forward(output, target)
        loss_nll = self.nll(output, target)
        
        loss = loss_cal + self.alpha * loss_nll
        return dict(loss=loss), loss_cal, loss_nll    

class MDCATrainLoss_alpha(MDCA):
    def __init__(self, alpha = 1, ignore_index=-1):
        super().__init__()
        self.nll = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.alpha = alpha
        print("Using alpha value = {}".format(self.alpha))

    def forward(self, output, target):
        # import pdb; pdb.set_trace()
        loss_cal = super().forward(output, target)
        loss_nll = self.nll(output, target)
        
        loss = loss_nll + self.alpha * loss_cal
        return dict(loss=loss), loss_cal, loss_nll    

class DCATrainLoss_alpha(DCA):
    def __init__(self, beta = 1, ignore_index=-1):
        super().__init__()
        self.nll = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.beta = beta
        print("Using beta value = {}".format(self.beta))

    def forward(self, output, target):
        # import pdb; pdb.set_trace()
        loss_cal = super().forward(output, target)
        loss_nll = self.nll(output, target)
        
        loss = loss_nll + self.beta * loss_cal
        return dict(loss=loss), loss_cal, loss_nll    


class MDCA_LabelSmoothLoss(nn.Module):
    def __init__(self, n_classes,alpha = 0, beta = 0, ignore_index=-1):
        super().__init__()
        self.n_classes = n_classes
        self.LSL = LabelSmoothingLoss(n_classes = self.n_classes , smoothing= alpha) 
        self.MDCA = MDCA()
        self.alpha = alpha
        self.beta = beta
        print("Using alpha value = {}".format(self.alpha))
        print("Using beta value = {}".format(self.beta))
    
    
    def reset(self):
        pass

    def forward(self, output, target):

        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image
        # outputs must be logits
        '''
        loss_cal = self.MDCA.forward(output, target)
        loss_nll = self.LSL(output, target)
        
        loss = loss_nll + self.beta * loss_cal
        return dict(loss=loss), loss_cal, loss_nll 

class MDCA_NLLLoss(nn.Module):
    def __init__(self, n_classes, beta = 0, ignore_index=-1):
        super().__init__()
        self.n_classes = n_classes
        self.CE = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.MDCA = MDCA()
        self.beta = beta
        print("Using beta value = {}".format(self.beta))
    
    
    def reset(self):
        pass

    def forward(self, output, target):

        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image
        # outputs must be logits
        '''
        loss_cal = self.MDCA.forward(output, target)
        loss_nll = self.CE(output, target)
        
        loss = loss_nll + self.beta * loss_cal
        return dict(loss=loss), loss_cal, loss_nll    

class ECETrainLoss_alpha(ECELoss):
    def __init__(self, n_classes, alpha, n_bins = 10, ignore_index=-1,**kwargs):
        super().__init__(n_classes, n_bins=n_bins)
        self.nll = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.alpha = alpha
        print("Using alpha value = {}".format(self.alpha))

    def forward(self, output, target):
        # import pdb; pdb.set_trace()
        loss_cal = super().forward(output, target)
        loss_nll = self.nll(output, target)
        
        loss = loss_cal + self.alpha * loss_nll
        return dict(loss=loss), loss_cal, loss_nll      

class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

        print("using gamma={}".format(gamma))

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    
    def reset(self):
        pass

class FocalMDCA_NLLLoss(nn.Module):
    def __init__(self, gamma=1.0, beta = 1.0, ignore_index=-1):
        super().__init__()
        
        self.beta = beta
        self.gamma = gamma

        self.CE = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) 
        self.MDCA = Focal_MDCA(gamma=self.gamma)
        
        print("Using beta value = {}".format(self.beta))
        print("Using gamma value = {}".format(self.gamma))
    
    def reset(self):
        pass

    def forward(self, output, target):
        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image
        # outputs must be logits
        '''
        loss_cal = self.MDCA.forward(output, target)
        loss_nll = self.CE(output, target)
        
        loss = loss_nll + self.beta * loss_cal
        return dict(loss=loss), loss_cal, loss_nll

class FocalMDCA_LS(nn.Module):
    def __init__(self, n_classes, alpha=0.1, gamma=1.0, beta = 1.0, ignore_index=-1):
        super().__init__()
        
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.n_classes = n_classes

        self.CE = LabelSmoothingLoss(n_classes = self.n_classes , smoothing= self.alpha) 
        self.MDCA = Focal_MDCA(gamma=self.gamma)
        
        print("Using beta value = {}".format(self.beta))
        print("Using gamma value = {}".format(self.gamma))
        print("Using alpha value = {}".format(self.alpha))
    
    def reset(self):
        pass

    def forward(self, output, target):
        '''
        output = [batch, n_Class] np array: The complete logit vector of an image 

        target = [batch] np array: The GT for the image
        # outputs must be logits
        '''
        loss_cal = self.MDCA.forward(output, target)
        loss_nll = self.CE(output, target)
        
        loss = loss_nll + self.beta * loss_cal
        return dict(loss=loss), loss_cal, loss_nll 