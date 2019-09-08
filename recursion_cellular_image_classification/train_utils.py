import sys

sys.path.append('/home/lyan/Documents/enorm/enorm')

PRINT_FREQ = 100

sys.path.append('/home/lyan/Documents/rxrx1-utils')

import torch
# from enorm import ENorm
import tqdm
import warnings

warnings.filterwarnings('ignore')

import time
from tqdm import *


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '  '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum / self.count)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def validate(loader, crit, model, metric):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':4.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader), losses, top1,
                             prefix='Test: ')

    model.eval()
    metric.eval()

    with torch.no_grad():
        end = time.time()
        with open('log.txt', 'w') as log_f:
            with tqdm(enumerate(loader), desc='validate', file=sys.stdout) as iterator:
                for i, (images, target) in iterator:
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    feature = model(images)
                    output = metric(feature, None)
                    loss = crit(output, target)

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    s = progress.print((i))
                    iterator.set_postfix_str(s)

    return top1.avg


def train(epoch, train_loader, model, metric, opt, crit, enorm):
    batch_time = AverageMeter('Time', ':4.3f')
    data_time = AverageMeter('Data', ':4.3f')
    losses = AverageMeter('Loss', ':4.3f')
    top1 = AverageMeter('Acc@1', ':4.2f')
    top5 = AverageMeter('Acc@5', ':4.2f')
    progress = ProgressMeter(len(train_loader), losses,
                             top1, prefix="Epoch: [{}]".format(epoch))
    i = 0

    model.train()
    metric.eval()
    end = time.time()
    with open('log.txt', 'w') as log_f:
        with tqdm(train_loader, desc='train', file=sys.stdout) as iterator:
            for data in iterator:
                opt.zero_grad()

                data_time.update(time.time() - end)

                data_input, target = data
                data_input = data_input.cuda()
                target = target.cuda()

                feature = model(data_input)
                output = metric(feature, target)
                loss = crit(output, target)

                opt.zero_grad()
                loss.backward()
                opt.step()

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), data_input.size(0))
                top1.update(acc1[0], data_input.size(0))
                top5.update(acc5[0], data_input.size(0))

                if enorm is not None:
                    enorm.step()

                batch_time.update(time.time() - end)
                end = time.time()

                s = progress.print(i)
                iterator.set_postfix_str(s)

                i += 1
    return top1.avg, losses.avg
