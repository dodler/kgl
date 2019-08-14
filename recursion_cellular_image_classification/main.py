import sys

PRINT_FREQ = 100

sys.path.append('/home/lyan/Documents/rxrx1-utils')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import pandas as pd

import torch.utils.data as D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import math

import tqdm

import warnings

warnings.filterwarnings('ignore')

from albumentations import (
    HorizontalFlip, ShiftScaleRotate, Compose
)

import cv2

import time
from tqdm import *
import pretrainedmodels as pm
import torchvision.transforms as transforms


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
        return '\t'.join(entries)

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


def validate(loader, crit, model):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        with open('log.txt','w') as log_f:
            with tqdm(loader, desc='validate', file=log_f) as iterator:
                for i, (images, target) in iterator:

                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    # compute output
                    output = model(images, torch.zeros_like(target))
                    loss = crit(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    s=progress.print((i))
                    iterator.set_postfix_str(s)

    return top1.avg


def train(epoch, train_loader, model, opt, crit, use_arc_metric=False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    i = 0

    model.train()
    end = time.time()
    with open('log.txt','w') as log_f:
        with tqdm(train_loader, desc='train', file=log_f) as iterator:
            for data in iterator:

                data_time.update(time.time() - end)

                data_input, target = data
                data_input = data_input.to(DEVICE)
                target = target.to(DEVICE)

                if use_arc_metric:
                    out = model(data_input, target)
                else:
                    out = model(data_input, None)

                loss = crit(out, target)
                acc1, acc5 = accuracy(out, target, topk=(1, 5))

                losses.update(loss.item(), data_input.size(0))
                top1.update(acc1[0], data_input.size(0))
                top5.update(acc5[0], data_input.size(0))

                opt.zero_grad()
                loss.backward()
                opt.step()

                batch_time.update(time.time() - end)
                end = time.time()

                s=progress.print(i)
                iterator.set_postfix_str(s)

                i += 1
    return top1.avg, losses.avg


def upd_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


class ImagesDS():
    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1, 2, 3, 4, 5, 6], aug=None):

        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.aug = aug
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.485, 0.456, 0.456, 0.406, 0.406],
                                              std=[0.229, 0.229, 0.224, 0.224, 0.225, 0.225])

    def _get_img_path(self, index, channel):
        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}', f'{well}_s{self.site}_w{channel}.png'])

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        target = {}
        for i in range(len(paths)):
            if i == 0:
                prefix = 'image'
            else:
                prefix = 'image' + str(i)
            img = cv2.imread(paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            target[prefix] = img

        augmented = self.aug(**target)
        img = np.zeros((512, 512, 6), dtype=np.uint8)
        k = list(augmented.keys())
        for i in range(len(augmented.keys())):
            img[:, :, i] = augmented[k[i]]

        img = self.to_tensor(img)
        #         img=self.normalize(img)

        if self.mode == 'train':
            label = torch.tensor(self.records[index].sirna, dtype=torch.int64)
        else:
            label = self.records[index].id_code

        return img, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


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


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=20.0, m=0.40, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ArcSEResnext50(nn.Module):
    def __init__(self, num_classes):
        super(ArcSEResnext50, self).__init__()
        self.feature_extr = pm.se_resnext50_32x4d()

        # self.feature_extr.load_state_dict(
        #     torch.load('../input/se-resnext-pytorch-pretrained/se_resnext50_32x4d-a260b3a4.pth'))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.metric = ArcMarginProduct(in_features=2048, out_features=num_classes)

    def upd_metric(self, metric):
        self.metric = metric

    def forward(self, x, labels):
        bs = x.shape[0]
        x = self.feature_extr.features(x)
        x = self.pool(x).reshape(bs, -1)

        if labels is not None:
            return self.metric(x, labels)
        else:
            return F.normalize(x)


model = ArcSEResnext50(1108)
# model.upd_metric(torch.nn.Linear(2048, 1108))

weights = torch.zeros(64, 6, 7, 7, dtype=torch.float32)
weights[:, 0, :, :] = model.feature_extr.layer0.conv1.weight[:, 0, :, :]
weights[:, 1, :, :] = model.feature_extr.layer0.conv1.weight[:, 0, :, :]
weights[:, 2, :, :] = model.feature_extr.layer0.conv1.weight[:, 1, :, :]
weights[:, 3, :, :] = model.feature_extr.layer0.conv1.weight[:, 1, :, :]
weights[:, 4, :, :] = model.feature_extr.layer0.conv1.weight[:, 2, :, :]
weights[:, 5, :, :] = model.feature_extr.layer0.conv1.weight[:, 2, :, :]

model.feature_extr.layer0.conv1 = nn.Conv2d(6, 64, (7, 7), (2, 2), (3, 3), bias=False)
model.feature_extr.layer0.conv1.weight = torch.nn.Parameter(weights)

model.to(0)


def augment_flips_color(p=.5, n=6):
    target = {}
    for i in range(n - 1):
        target['image' + str(i)] = 'image'

    return Compose([
        HorizontalFlip(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=10, p=.3),
    ], p=p, additional_targets=target)


path_data = '/var/ssd_1t/recursion_cellular_image_classification/'
device = 'cuda'
batch_size = 8

aug = augment_flips_color()
ds = ImagesDS(path_data + '/train.csv', img_dir=path_data, aug=aug, mode='train')
ds_test = ImagesDS(path_data + '/test.csv', path_data, mode='test')

loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
tloader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

LR = 1e-4
DEVICE = 0

for param in model.parameters():
    param.requires_grad = True

opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
crit = torch.nn.CrossEntropyLoss()


def save_state(model, opt, top1_avg, loss, name=None):
    state = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'top1_avg': top1_avg,
        'loss': loss
    }
    if name is None:
        save_name = 'last.pth'
    else:
        save_name = name
    torch.save(state, save_name)


best_score = 1 #validate(tloader, model)
for i in range(12):
    top1_avg, loss = train(i, loader, model, opt, crit=crit, use_arc_metric=True)
    # top1_avg = validate(tloader, crit=crit, model=model)
    # if top1_avg > best_score:
    #     save_state(model, opt, top1_avg, loss)
