import logging
import torch
import torch.nn as nn
import torch.optim as optim

from utils import AverageMeter, Logger
from solver.runners import test 
from calibration_library.ece_loss import ECELoss
from calibration_library.cce_loss import CCELossFast
from solver import loss_dict, EarlyStopping

import os

from tqdm import tqdm

def _freeze_model(model : nn.Module):
    for params in model.parameters():
        params.requires_grad = False

class Matrix_Scaling(nn.Module):
    def __init__(self, base_model, num_classes, optim='adam', reg="l2", Lambda=0.0, Mu=0.0):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        self.optim = optim
        self.Lambda = Lambda
        self.Mu = Mu
        self.reg = reg

        _freeze_model(self.base_model)

        self.setup_model()

    def setup_model(self):
        self.fc = nn.Linear(self.num_classes, self.num_classes)
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

    def regularizer(self):
        W, b = self.fc.parameters()

        if self.reg == "l2":
            return self.Lambda*((W**2).sum())
        elif self.reg == "odir":
            k = self.num_classes
            W, b = self.fc.parameters()

            # keep loss value 
            w_loss = ((W**2).sum() - (torch.diagonal(W, 0)**2).sum())/(k*(k-1))
            b_loss = ((b**2).sum())/k

            return self.Lambda*w_loss + self.Mu*b_loss

    def loss_func(self, outputs, targets):
        crit = nn.CrossEntropyLoss()
        return crit(outputs, targets) + self.regularizer()

    def give_params(self):
        return self.fc.parameters()

    def fit(self, train_loader, lr=0.001, epochs=500, patience=15):

        if self.optim == "sgd":
            optimizer = optim.SGD(self.give_params(), 
                                lr=lr,
                                weight_decay=0.0)

        elif self.optim == "adam":
            optimizer = optim.Adam(self.give_params(),
                                lr=lr,
                                weight_decay=0.0)
        
        scheduler = EarlyStopping(patience=patience)

        # send model to gpu
        self.cuda()

        last_loss = 0.0

        bar = tqdm(range(epochs))
        for i in bar:
            avg_loss = AverageMeter()
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                imgs, labels = imgs.cuda(), labels.cuda()

                outs = self.forward(imgs)
                loss = self.loss_func(outs, labels)

                loss.backward()
                optimizer.step()

                avg_loss.update(loss.item())
            
            last_loss = avg_loss.avg
            bar.set_postfix_str("loss : {:.5f} | lr : {:.5f}".format(avg_loss.avg, lr))
            if scheduler.step(avg_loss.avg):
                break

        print("Calibration Done!!")
        return last_loss 
    
    def calibrate(self, train_loader, lr=0.001, epochs=500, double_fit=True, patience=15):

        loss = self.fit(train_loader, lr, epochs, patience)

        if double_fit:
            print("Trying to double fit...")
            lr /= 10
            loss = self.fit(train_loader, lr, epochs, patience)
        
        return loss


class Dirichilit_Calibration(Matrix_Scaling):
    def __init__(self, base_model, num_classes, optim='adam', reg="odir", Lambda=0.0, Mu=0.0):
        super().__init__(base_model, num_classes, optim, reg, Lambda, Mu)
    
    def forward(self, x):
        # print("hello")
        x = self.base_model(x)
        x = torch.log_softmax(x, dim=1)
        x = self.fc(x)
        return x

def calibrate_dir(model_save_folder, model, train_loader, test_loader, num_classes, lr=0.001, epochs=500, regularizer="l2", optim='adam', double_fit=True, patience=15, Lambdas=[0.0], Mus=[0.0]):

    print("Calibration of model with Dirichilit Calibration...")
    print("using regularizer =", regularizer)

    print("Lambdas:", Lambdas)
    print("Mus:", Mus)

    if regularizer == "l2":

        path_to_logger = os.path.join(model_save_folder, 'l2_dirichlet.txt')
        res = False
        if os.path.isfile(path_to_logger):
            res = True
        logger = Logger(path_to_logger, resume=res)  
        logger.set_names(['lambda', 'val_loss', 'top1', 'top3', 'top5', 'SCE', 'ECE'])
        for l in Lambdas:
            print("Now running for Lambda={:.5f}".format(l))
            calib_model = Dirichilit_Calibration(model, num_classes, optim, regularizer, Lambda=l)
            calib_model.calibrate(train_loader, lr, epochs, double_fit, patience)
            
            nll_loss, top1, top3, top5, SCE, ECE = calib_metrics(calib_model, test_loader)
            logger.append([l, nll_loss, top1, top3, top5, SCE, ECE])
    
    assert regularizer == 'odir'
    path_to_logger = os.path.join(model_save_folder, 'odir_dirichlet.txt')
    res = False
    if os.path.isfile(path_to_logger):
        res = True
    logger = Logger(path_to_logger, resume=res) 
    logger.set_names(['lambda', 'mu', 'val_loss', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

    for l,m in zip(Lambdas, Mus):
        print("Now running for Lambda={:.5f} | Mu={:.5f}".format(l, m))
        calib_model = Dirichilit_Calibration(model, num_classes, optim, regularizer, Lambda=l, Mu=m)
        calib_model.calibrate(train_loader, lr, epochs, double_fit, patience)

        nll_loss, top1, top3, top5, SCE, ECE = calib_metrics(calib_model, test_loader)
        logger.append([l, m, nll_loss, top1, top3, top5, SCE, ECE])

def calibrate_mat(model_save_folder, model, train_loader, test_loader, num_classes, lr=0.001, epochs=500, regularizer="l2", optim='adam', double_fit=True, patience=15, Lambdas=[0.0], Mus=[0.0]):

    print("Calibration of model with Matrix Scaling...")
    print("using regularizer =", regularizer)

    print("Lambdas:", Lambdas)
    print("Mus:", Mus)

    if regularizer == "l2":

        path_to_logger = os.path.join(model_save_folder, 'l2_dirichlet.txt')
        res = False
        if os.path.isfile(path_to_logger):
            res = True
        logger = Logger(path_to_logger, resume=res)  
        logger.set_names(['lambda', 'val_loss', 'top1', 'top3', 'top5', 'SCE', 'ECE'])
        for l in Lambdas:
            print("Now running for Lambda={:.5f}".format(l))
            calib_model = Matrix_Scaling(model, num_classes, optim, regularizer, Lambda=l)
            calib_model.calibrate(train_loader, lr, epochs, double_fit, patience)
            
            nll_loss, top1, top3, top5, SCE, ECE = calib_metrics(calib_model, test_loader)
            logger.append([l, nll_loss, top1, top3, top5, SCE, ECE])
    
    assert regularizer == 'odir'
    path_to_logger = os.path.join(model_save_folder, 'odir_dirichlet.txt')
    res = False
    if os.path.isfile(path_to_logger):
        res = True
    logger = Logger(path_to_logger, resume=res) 
    logger.set_names(['lambda', 'mu', 'val_loss', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

    for l,m in zip(Lambdas, Mus):
        print("Now running for Lambda={:.5f} | Mu={:.5f}".format(l, m))
        calib_model = Matrix_Scaling(model, num_classes, optim, regularizer, Lambda=l, Mu=m)
        calib_model.calibrate(train_loader, lr, epochs, double_fit, patience)

        nll_loss, top1, top3, top5, SCE, ECE = calib_metrics(calib_model, test_loader)
        logger.append([l, m, nll_loss, top1, top3, top5, SCE, ECE])


def calib_metrics(calib_model, test_loader):
    """returns the nll_loss, top1, top3, top5, SCE, ECE score"""
    # set up metrics
    ece_criterion = ECELoss(calib_model.num_classes)    
    sce_criterion = CCELossFast(calib_model.num_classes)
    crit = loss_dict["NLL"]()

    print("Running calibrated model on test set...")
    nll_loss, top1, top3, top5, SCE, ECE = test(test_loader, calib_model, crit, ece_criterion, sce_criterion)
    print("Results after Calibration on test set:")
    print("top1 : {:.5f} | top3 : {:.5f} | top5 : {:.5f} | SCE : {:.5f} | ECE : {:.5f}".format(
            top1, top3, top5, SCE, ECE
        )
    )
    print()
    
    return nll_loss, top1, top3, top5, SCE, ECE