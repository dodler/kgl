import argparse
import datetime
import sys

from torchcontrib.optim import SWA

from recursion_cellular_image_classification.cell_data import ImagesDS
from recursion_cellular_image_classification.models import ArcEffNetb0, ArcMarginProduct
from recursion_cellular_image_classification.train_utils import train, validate

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)

sys.path.append('/home/lyan/Documents/enorm/enorm')

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

PRINT_FREQ = 100

sys.path.append('/home/lyan/Documents/rxrx1-utils')

import torch.utils.data as D

import torch
import torch.nn as nn
# from enorm import ENorm
import warnings

warnings.filterwarnings('ignore')

from albumentations import (
    HorizontalFlip, ShiftScaleRotate, Compose,
    IAAAdditiveGaussianNoise)

parser = argparse.ArgumentParser(description='SIIM ACR Pneumotorax unet training')
parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--opt', default='SGD', choices=['SGD', 'Adam', 'Adamw'], type=str)
parser.add_argument('--batch-size', default=36, type=int)
parser.add_argument('--num-workers', default=14, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--resume', required=False, type=str, default=None)
args = parser.parse_args()


def upd_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


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


metric = ArcMarginProduct(in_features=1280, out_features=1108)
metric.train()

model = ArcEffNetb0()
model.train()

if args.resume is not None:
    print('resuming from', args.resume)
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    metric.load_state_dict(ckpt['metric'])

crit = torch.nn.CrossEntropyLoss()
# enorm = ENorm(model.feature_extr.named_parameters(), opt, model_type='conv', c=1)
enorm = None

if torch.cuda.is_available():
    metric.cuda()
    model.cuda()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    metric = torch.nn.DataParallel(metric)

opt = torch.optim.SGD([{'params': model.parameters()}, {'params': metric.parameters()}],
                      lr=args.lr, weight_decay=5e-4, momentum=0.9)
opt = SWA(opt, swa_start=20, swa_freq=10, swa_lr=args.lr / 2)

model.cuda()


def augment_flips_color(p=.5, n=6):
    target = {}
    for i in range(n - 1):
        target['image' + str(i)] = 'image'

    return Compose([
        HorizontalFlip(),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=10, p=.3),
    ], p=p, additional_targets=target)


def nothing(p=.5, n=6):
    target = {}
    for i in range(n - 1):
        target['image' + str(i)] = 'image'

    return Compose([], p=p, additional_targets=target)


path_data = '/var/ssd_1t/recursion_cellular_image_classification/'
device = 'cuda'
batch_size = args.batch_size

aug = augment_flips_color()
ds_train = ImagesDS(path_data + 'split_train.csv', img_dir=path_data, aug=aug, mode='train')
ds_valid = ImagesDS(path_data + 'valid.csv', path_data, mode='train', aug=nothing())

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size,
                                           shuffle=True, num_workers=args.num_workers)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size,
                                           shuffle=False, num_workers=args.num_workers)


def save_state(model, metric, opt, top1_avg, loss, epoch, name=None):
    if torch.cuda.device_count() > 1:
        state = {
            'metric': metric.module.state_dict(),
            'model': model.module.state_dict(),
            'opt': opt.state_dict(),
            'top1_avg': top1_avg,
            'loss': loss,
            'epoch': epoch
        }
    else:
        state = {
            'metric': metric.module.state_dict(),
            'model': model.module.state_dict(),
            'opt': opt.state_dict(),
            'top1_avg': top1_avg,
            'loss': loss,
            'epoch': epoch
        }
    if name is None:
        save_name = '/var/data/protein_checkpoints/last.pth'
    else:
        save_name = name
    torch.save(state, save_name)


scheduler = ReduceLROnPlateau(opt, patience=20, mode='max', min_lr=1e-6, verbose=True)

if args.resume is not None:
    start_epoch = ckpt['epoch'] + 1
else:
    start_epoch = 0

best_score = 0
for i in range(start_epoch, args.epochs):
    top1_avg, loss = train(i, train_loader, model, metric, opt, crit=crit, enorm=enorm)
    scheduler.step(top1_avg, i)
    top1_avg = validate(valid_loader, crit=crit, model=model, metric=metric)
    print('val score', top1_avg)
    if i > 10:
        opt.swap_swa_sgd()
    if top1_avg > best_score:
        ckpt_name = '/var/data/checkpoints_protein/arcface_effnet_b0_epoch' + str(
            i) + '_' + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H_%M_%S") + '_.pth'
        save_state(model, metric, opt, top1_avg, loss,
                   epoch=i,
                   name=ckpt_name)
