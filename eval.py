import argparse
import os
import shutil
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils import Logger, AverageMeter, accuracy, mkdir_p, parse_args
from utils import get_lr, save_checkpoint
from tqdm import tqdm

from solver.runners import test, get_logits_targets
from solver import loss_dict

from calibration_library.ece_loss import ECELoss
from calibration_library.cce_loss import CCELossFast

from models import model_dict
from data_loaders import dataloader_dict, dataset_nclasses_dict, dataset_class_dict
from tools.visualisation.misclassification import saver

import logging

if __name__ == "__main__":
    
    torch.manual_seed(1)
    random.seed(1)
    
    args = parse_args()

    assert args.dataset in dataloader_dict
    assert args.model in model_dict
    assert args.dataset in dataset_nclasses_dict
    assert args.dataset in dataset_class_dict


    classes = dataset_class_dict[args.dataset]
    num_classes = len(classes)
    # prepare model
    model = model_dict[args.model](num_classes=num_classes)
    model.cuda()

    # set up dataset
    train_loader, val_loader = dataloader_dict[args.dataset](args)

    logging.info(f"Using dataset : {args.dataset}")
    logging.info(f"Using loss function : NLL")

    # set up metrics

    assert args.resume, "Please provide a trained model file"
    assert os.path.isfile(args.resume)
    # Load checkpoint.
    # linux agnostic
    prefix = os.path.dirname(args.resume).split("/")[-2]+ os.path.dirname(args.resume).split("/")[-1]
    # print(prefix)
    newSaver = saver(prefix, args.dataset, classes)


    logging.info(f'Resuming from saved checkpoint: {args.resume}')
   
    model_save_path = os.path.dirname(args.resume)

    saved_model_dict = torch.load(args.resume)

    # assert args.model == saved_model_dict['model']
    # assert args.dataset == saved_model_dict['dataset']

    model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()
    
    logits, targets = get_logits_targets(val_loader, model)
    logits = logits.softmax(dim =1)
    newSaver.batch_forward(logits.detach().cpu().numpy(), targets.detach().cpu().numpy())
    print(logits.shape)
    print(targets.shape)

