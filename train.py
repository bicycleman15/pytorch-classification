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

from solver.runners import train, test
from solver import loss_dict

from calibration_library.ece_loss import ECELoss
from calibration_library.cce_loss import CCELossFast

from models import model_dict
from data_loaders import dataloader_dict, dataset_nclasses_dict

import logging

def create_loss_save_str(args):
    loss_name = args.lossname

    save_str = loss_name

    if "LS" in loss_name:
        save_str += f"_alpha={args.alpha}"
    
    if "DCA" in loss_name:
        save_str += f"_beta={args.beta}"
    
    if "FL" in loss_name:
        save_str += f"_gamma={args.gamma}"
    
    return save_str

if __name__ == "__main__":
    
    torch.manual_seed(1)
    random.seed(1)
    
    args = parse_args()

    assert args.dataset in dataloader_dict
    assert args.model in model_dict
    assert args.lossname in loss_dict
    assert args.dataset in dataset_nclasses_dict

    loss_save_string = create_loss_save_str(args)
    orig_loss_string = loss_save_string

    if len(args.prefix):
        loss_save_string = args.prefix + "-" + loss_save_string

    # prepare save path
    model_save_pth = f"{args.checkpoint}/{args.dataset}/{args.model}/{loss_save_string}"
    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    logging.basicConfig(level=logging.DEBUG, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")

    num_classes = dataset_nclasses_dict[args.dataset]
    criterion = loss_dict[args.lossname](alpha=args.alpha, beta=args.beta, gamma=args.gamma, n_classes=num_classes)

    logging.info(f"Using loss function : {orig_loss_string}")
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, testloader = dataloader_dict[args.dataset](args)


    # set up metrics
    ece_evaluator = ECELoss(n_classes = num_classes)    
    fastcce_evaluator = CCELossFast(n_classes = num_classes)

    logging.info(f"Setting up optimizer : {args.optimizer}")

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), 
                              lr=args.lr, 
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    
    logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    start_epoch = args.start_epoch

    if args.resume:
        # Load checkpoint.

        logging.info(f'Resuming from saved checkpoint: {args.resume}')

        assert os.path.isfile(args.resume)
        args.checkpoint = os.path.dirname(args.resume)

        saved_model_dict = torch.load(args.resume)
        start_epoch = saved_model_dict['epoch']
        model.load_state_dict(saved_model_dict['state_dict'])
        optimizer.load_state_dict(saved_model_dict['optimizer'])
        scheduler.load_state_dict(saved_model_dict['scheduler'])
        
        model.cuda()
    
    best_acc = 0.

    # set up loggers
    logger = Logger(os.path.join(model_save_pth, 'train_metrics.txt')) 
    logger.set_names(['lr', 'train_loss', 'val_loss', 'top1_train', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

    for epoch in range(start_epoch, args.epochs):

        logging.info('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, get_lr(optimizer)))

        train_loss, top1_train, _, _ = train(trainloader, model, criterion, optimizer)
        test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, model, criterion, ece_evaluator, fastcce_evaluator)

        scheduler.step()

        # append logger file
        logger.append([get_lr(optimizer), train_loss, test_loss, top1_train, top1, top3, top5, cce_score, ece_score])

        logging.info("End of epoch {} stats: train_loss : {:.4f} | val_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f} | ECE : {:.5f} | SCE : {:.5f}".format(
            epoch+1,
            train_loss,
            test_loss,
            top1_train,
            top1,
            ece_score,
            cce_score
        ))

        # save model
        is_best = top1 > best_acc
        best_acc = max(top1, best_acc)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': top1,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'dataset' : args.dataset,
                'model' : args.model
            }, is_best, checkpoint=model_save_pth)

    # DO UMAP T_SNE ....
    logger.close()

    logging.info('Best accuracy obtained: {}'.format(best_acc))