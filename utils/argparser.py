import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training for calibration')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar100', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--train-batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch-size', default=100, type=int, metavar='N',
                        help='test batchsize')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--alpha', default=0.1, type=float,
                        metavar='ALPHA', help='alpha to train Label Smoothing with')
    parser.add_argument('--beta', default=10, type=float,
                        metavar='BETA', help='beta to train DCA/MDCA with')
    parser.add_argument('--gamma', default=1, type=float,
                        metavar='GAMMA', help='gamma to train Focal Loss with')

    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')

    parser.add_argument('--schedule-steps', type=int, nargs='+', default=[150, 225],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, help='LR is multiplied by this on schedule.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--lossname', default='NLL', type=str, metavar='LNAME')
    parser.add_argument('--model', default='resnet34', type=str, metavar='MNAME')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='ONAME')

    parser.add_argument('--prefix', default='', type=str, metavar='PRNAME')
    
    return parser.parse_args()