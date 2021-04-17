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
from data_loaders import temploader_dict, dataset_nclasses_dict

import logging

if __name__ == "__main__":
    
    torch.manual_seed(1)
    random.seed(1)
    
    args = parse_args()

    assert args.dataset in temploader_dict
    assert args.model in model_dict
    assert args.dataset in dataset_nclasses_dict

    logging.basicConfig(level=logging.DEBUG, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.StreamHandler()
                        ])

    criterion = loss_dict["NLL"]()

    num_classes = dataset_nclasses_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    model.cuda()

    # set up dataset
    temp_loader, val_loader = temploader_dict[args.dataset](args)

    logging.info(f"Using dataset : {args.dataset}")
    logging.info(f"Using loss function : NLL")

    # set up metrics
    ece_criterion = ECELoss(n_classes = num_classes)    
    sce_criterion = CCELossFast(n_classes = num_classes)

    assert args.resume, "Please provide a trained model file"
    assert os.path.isfile(args.resume)
    # Load checkpoint.

    logging.info(f'Resuming from saved checkpoint: {args.resume}')
   
    model_save_path = os.path.dirname(args.resume)

    saved_model_dict = torch.load(args.resume)

    assert args.model == saved_model_dict['model']
    assert args.dataset == saved_model_dict['dataset']

    model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()
    
    nll_score = float('inf')
    sce_score = float('inf')
    ece_score = float('inf')
    best_temp = 0.

    logging.info("getting logits...")
    logits, targets = get_logits_targets(temp_loader, model, criterion)

    logging.info("Now running temp scaling...")

    T = 0.1

    for i in (range(100)):

        outputs = logits / T

        ece_criterion.reset()
        sce_criterion.reset()

        cur_nll_score = criterion(outputs, targets)[0]['loss'].item() # Our loss returns a dict
        ece_criterion.forward(outputs, targets)
        sce_criterion.forward(outputs, targets)

        ECE = ece_criterion.get_overall_ECELoss().item()
        SCE = sce_criterion.get_overall_CCELoss().item()

        if cur_nll_score < nll_score:
            nll_score = cur_nll_score
            sce_score = SCE
            ece_score = ECE
            best_temp = T

        # print(T, cur_nll_score, ECE, SCE)

        T += 0.1

    logging.info("Best Temp Found, running on val set...")
    test_loss, top1, top3, top5, sce_score, ece_score = test(val_loader, model, criterion, ece_criterion, sce_criterion, T=best_temp)

    logging.info("Temp Scaling Done...")
    logging.info("Best T={} | SCE : {:.5f} | ECE : {:.5f}".format(best_temp, sce_score, ece_score))