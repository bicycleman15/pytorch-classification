import torch
from tqdm import tqdm
from utils import Logger, AverageMeter, accuracy


def train(trainloader, model, criterion, optimizer):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    criterion.reset()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in bar:
        
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)

        loss_dict = criterion(outputs, targets)

        loss = loss_dict[0]["loss"]

        # measure accuracy and record loss
        prec1, prec3, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))

        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))

    return (losses.avg, top1.avg, top3.avg, top5.avg)

@torch.no_grad()
def test(testloader, model, criterion, ece_criterion, sce_criterion, T=1.0):

    criterion.reset()
    ece_criterion.reset()
    sce_criterion.reset()

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        outputs /= T

        loss_dict = criterion(outputs, targets)

        loss = loss_dict[0]["loss"]

        ece_criterion.forward(outputs,targets)
        sce_criterion.forward(outputs,targets)
        
        prec1, prec3, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))


        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))

    eces = ece_criterion.get_overall_ECELoss()
    cces = sce_criterion.get_overall_CCELoss()

    return (losses.avg, top1.avg, top3.avg, top5.avg, cces.item(), eces.item())

@torch.no_grad()
def get_logits_targets(testloader, model, criterion):

    criterion.reset()

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_targets = None
    all_outputs = None

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in bar:

        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = torch.cat([all_targets, targets], dim=0)
            all_outputs = torch.cat([all_outputs, outputs], dim=0)

    return all_outputs, all_targets
