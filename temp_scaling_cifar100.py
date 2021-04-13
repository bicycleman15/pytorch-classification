'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from solver.loss import CCETrainLoss_alpha, MDCATrainLoss_alpha, MDCA_LabelSmoothLoss, DCATrainLoss_alpha, MDCA_NLLLoss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from resnet import cifar_resnet56
from cifarc_dataloader import CIFAR100_C
from tqdm import tqdm

from calibration_library.ece_loss import ECELoss
from calibration_library.cce_loss import CCELossFast

from torch.utils.tensorboard import SummaryWriter
import umap


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--alpha', default=0.1, type=float,
                    metavar='ALPHA', help='alpha to train with')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lossname', default='', type=str, metavar='LNAME')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataloader = datasets.CIFAR100
    num_classes = 100
    classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

    ece_evaluator = ECELoss(n_classes = num_classes)
    fastcce_evaluator = CCELossFast(n_classes = num_classes)

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    
    sub_testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    full_testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    # testset = CIFAR100_C(severity=3, transform=transform_test)

    indices = list(range(0, len(full_testset)))
    random.shuffle(indices)
    ss = torch.utils.data.sampler.SubsetRandomSampler(indices[:int(0.1 * len(full_testset))])

    testloader = data.DataLoader(full_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    subloader = data.DataLoader(sub_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, sampler=ss)

    # Model
    print("==> creating model arch {} with depth={}".format(args.arch, args.depth))
    if args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    model = model.cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    assert args.resume
    
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    prefix = args.checkpoint
    title = prefix
    checkpoint = torch.load(args.resume)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.cuda()
    
    print('\nEvaluation only')

    best_val_loss = float('inf')
    best_t = -1

    for t in tqdm([1.0, 1.2, 1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]):
        train_loss, top1, top3, top5, cce_score_t, ece_score_t = test(subloader, model, criterion, 0, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix, t)
        
        print(f"T={t} | SCE={cce_score_t} | ECE={ece_score_t}")

        if(train_loss < best_val_loss):
            best_t = t
            best_val_loss = train_loss


    test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, model, criterion, 0, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix, best_t)
    print(f"Best T={best_t}  SCE={cce_score}   ECE={ece_score}")

    # run t-sne last
    # tsne(testloader, model, criterion, start_epoch, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix)
    # do_umap(testloader, model, criterion, start_epoch, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix)

@torch.no_grad()
def test(testloader, model, criterion, epoch, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix, T=1.0):
    global best_acc

    # criterion.reset()
    ece_evaluator.reset()
    fastcce_evaluator.reset()

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)

        outputs /= T

        loss_dict = criterion(outputs, targets)

        if type(loss_dict) == type((1, 2)):
            loss = loss_dict[0]["loss"]
        else:
            loss = loss_dict

        # measure accuracy and record loss
        with torch.no_grad():
            ece_evaluator.forward(outputs,targets)
            fastcce_evaluator.forward(outputs,targets)
        
        prec1, prec3, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        end = time.time()
    # bar.finish()

    eces = ece_evaluator.get_overall_ECELoss()
    cces = fastcce_evaluator.get_overall_CCELoss()

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces, eces)

@torch.no_grad()
def tsne(testloader, model, criterion, epoch, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix):
    global best_acc

    # switch to evaluate mode
    model.eval()

    overall_outputs = []
    overall_targets = []


    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:
        overall_targets.append(targets.cpu().numpy())
        inputs, targets = inputs.cuda(), targets.cuda()
        # compute output
        outputs = model(inputs).softmax(dim =1).cpu().numpy()
        overall_outputs.append(outputs)

    overall_outputs = np.array(overall_outputs).reshape(-1, len(classes))
    overall_targets = np.array(overall_targets).reshape(-1,1).astype(np.int)
    overall_targets_labels =[]
    # for i in range(overall_targets.shape[0]):
        # overall_targets_labels.append(classes[int(overall_targets[i])])
    # print(overall_outputs.shape)
    # print(overall_targets.shape)

    # overall_targets_labels = np.array(overall_targets_labels).reshape(-1,1)

    tsne_model = TSNE(random_state=0)

    tsne_data = tsne_model.fit_transform(overall_outputs)
    # print(tsne_data.shape)

    tsne_data = np.hstack((tsne_data, overall_targets))
    # print(tsne_data.shape)

    tsne_df = pd.DataFrame(data = tsne_data, columns = ("Dim1", "Dim2", "Labels"))
    g =sn.FacetGrid(tsne_df, hue = "Labels", size = 6)
    # sn.FacetGrid(tsne_df, hue = "Labels", size = 6).add_legend().map(plt.scatter, "Dim1", "Dim2")

    g.map_dataframe(sn.scatterplot, x="Dim1", y="Dim2")
    g.set_axis_labels("Dim1", "Dim2")
    # g.add_legend()
    plt.legend(classes)

    plt.savefig(f"t-sne-plots/{prefix}.jpeg")

    # import cv2
    # tsne_img = cv2.imread(f"t-sne-plots/{prefix}.jpeg")
    # writer.add_image("t-SNE Plot", tsne_img, epoch, dataformats="HWC")
    # writer.add_scalar("TB testing", epoch, 1)

    # writer.flush()

@torch.no_grad()
def do_umap(testloader, model, criterion, epoch, use_cuda, ece_evaluator, fastcce_evaluator, classes, prefix):
    global best_acc

    # switch to evaluate mode
    model.eval()

    overall_outputs = []
    overall_targets = []


    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:
        overall_targets.append(targets.cpu().numpy())
        inputs, targets = inputs.cuda(), targets.cuda()
        # compute output
        outputs = model.give_second_last(inputs).cpu().numpy()
        feat = outputs.shape[1]
        overall_outputs.append(outputs)

    overall_outputs = np.array(overall_outputs).reshape(-1, feat)
    overall_targets = np.array(overall_targets).reshape(-1,1).astype(np.int)
    overall_targets_labels = []
    # for i in range(overall_targets.shape[0]):
        # overall_targets_labels.append(classes[int(overall_targets[i])])
    print(overall_outputs.shape)
    print(overall_targets.shape)

    # overall_targets_labels = np.array(overall_targets_labels).reshape(-1,1)

    n_neighbors=5
    min_dist=0.3

    tsne_data = umap.UMAP(n_neighbors=n_neighbors,
                      min_dist=min_dist,
                      metric='euclidean').fit_transform(overall_outputs)

    print(tsne_data.shape)

    tsne_data = np.hstack((tsne_data, overall_targets))
    # print(tsne_data.shape)

    tsne_df = pd.DataFrame(data = tsne_data, columns = ("Dim1", "Dim2", "Labels"))
    g =sn.FacetGrid(tsne_df, hue = "Labels", size = 6)
    # sn.FacetGrid(tsne_df, hue = "Labels", size = 6).add_legend().map(plt.scatter, "Dim1", "Dim2")

    g.map_dataframe(sn.scatterplot, x="Dim1", y="Dim2")
    g.set_axis_labels("Dim1", "Dim2")
    # g.add_legend()
    plt.legend(classes)

    name_to_save = f"{prefix}-nn={n_neighbors}-mind={min_dist}.jpeg"
    plt.savefig(f"umap-plots/"+name_to_save)

    # import cv2
    # tsne_img = cv2.imread(f"umap-plots/"+name_to_save)
    # writer.add_image("UMAP Plot", tsne_img, epoch, dataformats="HWC")
    # writer.add_scalar("TB testing", epoch, 1)

    # writer.flush()

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
