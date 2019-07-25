from __future__ import print_function, division, absolute_import

import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import *

tqdm = tqdm_notebook


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None

    assert rater_a.shape[0] == rater_b.shape[0], 'y and y_pred shape mismatch'
    if min_rating is None:
        min_rating = min(np.min(rater_a), np.min(rater_b))
    if max_rating is None:
        max_rating = max(np.max(rater_a), np.max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = np.power(i - j, 2.0) / np.power(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


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
        print('\t'.join(entries))

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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def validate(model, valid_loader, crit):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    qwk = AverageMeter('QWK', ':6.2f')
    progress = ProgressMeter(len(valid_loader), batch_time, losses, top1, qwk,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(valid_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images, torch.zeros_like(target))
            loss = crit(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            pred = output.detach().cpu().numpy()
            pred = np.argmax(pred, axis=1)
            qwk_score = float(quadratic_weighted_kappa(pred, target.detach().cpu().numpy().reshape(-1)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            qwk.update(qwk_score, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} qwk {qwk.avg:.3f}'
              .format(top1=top1, qwk=qwk))

    return top1.avg


def train(epoch, train_loader, model, opt, crit, device=0, use_arc_metric=False, accumulate=0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    qwk = AverageMeter('QWK', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             qwk, prefix="Epoch: [{}]".format(epoch))
    i = 0

    batch_loss=0

    model.train()
    end = time.time()
    for data in tqdm(train_loader):

        data_time.update(time.time() - end)

        data_input, target = data
        data_input = data_input.to(device)
        target = target.to(device)

        if use_arc_metric:
            out = model(data_input, target)
        else:
            out = model(data_input, None)

        loss = crit(out, target)

        if accumulate<1:
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            loss.backward()
            batch_loss+=loss.item()
            if cnt%accumulate==0:
                opt.step()
                opt.zero_grad()
                batch_loss=0

        acc1 = accuracy(out, target, topk=(1,))
        pred = out.detach().cpu().numpy()
        pred = np.argmax(pred, axis=1)
        qwk_score = float(quadratic_weighted_kappa(pred, target.detach().cpu().numpy().reshape(-1)))

        losses.update(loss.item(), data_input.size(0))
        top1.update(acc1[0].item(), data_input.size(0))
        qwk.update(qwk_score, data_input.size(0))


        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.print(i)

        i += 1


def upd_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr
