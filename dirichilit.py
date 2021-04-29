import os
import random
import torch
import numpy as np

from utils import parse_args
from calibration_library.calibrators import calibrate_dir

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

    num_classes = dataset_nclasses_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    model.cuda()

    # set up dataset
    temp_loader, val_loader = temploader_dict[args.dataset](args)

    logging.info(f"Using dataset : {args.dataset}")
    logging.info(f"Using loss function : NLL")

    assert args.resume, "Please provide a trained model file"
    assert os.path.isfile(args.resume)
    # Load checkpoint.

    logging.info(f'Resuming from saved checkpoint: {args.resume}')
   
    model_save_folder = os.path.dirname(args.resume)
    saved_model_dict = torch.load(args.resume)

    if "dataset" in saved_model_dict:
        assert args.model == saved_model_dict['model']
        assert args.dataset == saved_model_dict['dataset']

    model.load_state_dict(saved_model_dict['state_dict'])

    # 10 classes start L2 parameters from -5.0, 100 classes startfrom -2.0   
    if args.dataset == "cifar100":
        start_from = -2.0
    else:
        start_from = -5.0

    # Set regularisation parameters to check through
    lambdas = np.array([10**i for i in np.arange(start_from, 7)])
    lambdas = sorted(np.concatenate([lambdas, lambdas*0.25, lambdas*0.5]))         
    mus = np.array([10**i for i in np.arange(start_from, 7)])
    
    lambdas = list(lambdas)
    mus = list(mus)

    calibrate_dir(model_save_folder, model, temp_loader, val_loader, num_classes, args.lr, args.epochs, 
        regularizer=args.regularizer,
        Lambdas=lambdas, Mus=mus, optim=args.optimizer
    )