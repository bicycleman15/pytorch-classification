import os
import pickle
import random
import torch

from utils import parse_args

from solver.runners import get_logits_targets

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

    # Load checkpoint --------
    assert args.resume, "Please provide a trained model file"
    assert os.path.isfile(args.resume)
    logging.info(f'Resuming from saved checkpoint: {args.resume}')
    model_save_folder = os.path.dirname(args.resume)
    saved_model_dict = torch.load(args.resume)
    if "model" in saved_model_dict:
        assert args.model == saved_model_dict['model']
        assert args.dataset == saved_model_dict['dataset']
    model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()


    logging.info("getting logits...")
    val_logits, val_targets = get_logits_targets(temp_loader, model)
    test_logits, test_targets = get_logits_targets(val_loader, model)

    with open(os.path.join(model_save_folder, "precalibrated_valtest_logits.p"), "wb") as f:
    # with open(os.path.join("precalibrated_valtest_logits.p"), "wb") as f:
        pickle.dump(((val_logits, val_targets), (test_logits, test_targets)), f)
