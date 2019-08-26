import argparse
import sys

from torchcontrib.optim import SWA

from recursion_cellular_image_classification.cell_data import ImagesDS
from recursion_cellular_image_classification.models import ArcEffNetb0
from recursion_cellular_image_classification.train_utils import train, validate

sys.path.append('/home/lyan/Documents/enorm/enorm')

from torch.optim.lr_scheduler import ReduceLROnPlateau

PRINT_FREQ = 100

sys.path.append('/home/lyan/Documents/rxrx1-utils')

import torch.utils.data as D

import torch
import torch.nn as nn
# from enorm import ENorm
import warnings

warnings.filterwarnings('ignore')

from albumentations import (
    HorizontalFlip, ShiftScaleRotate, Compose
)

parser = argparse.ArgumentParser(description='SIIM ACR Pneumotorax unet training')
parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--opt', default='SGD', choices=['SGD', 'Adam', 'Adamw'], type=str)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--resume', required=False, type=str)
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


model = ArcEffNetb0(1108)
model.train()
for p in model.parameters():
    p.requires_grad = True

for param in model.parameters():
    param.requires_grad = True

opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
crit = torch.nn.CrossEntropyLoss()
# enorm = ENorm(model.feature_extr.named_parameters(), opt, model_type='conv', c=1)
enorm = None
opt = SWA(opt, swa_start=10, swa_freq=5, swa_lr=1e-2)

if args.resume is not None:
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    # opt.load_state_dict(ckpt['opt'])

model = torch.nn.DataParallel(model)

model.cuda()


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
batch_size = args.batch_size

aug = augment_flips_color()
ds_train = ImagesDS(path_data + 'split_train.csv', img_dir=path_data, aug=aug, mode='train')
ds_valid = ImagesDS(path_data + 'valid.csv', path_data, mode='train')

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=8)

LR = 1e-4
DEVICE = 0


def save_state(model, opt, top1_avg, loss, name=None):
    state = {
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'top1_avg': top1_avg,
        'loss': loss
    }
    if name is None:
        save_name = '/var/data/protein_checkpoints/last.pth'
    else:
        save_name = name
    torch.save(state, save_name)


scheduler = ReduceLROnPlateau(opt, verbose=True, mode='min', patience=150)

# best_score = validate(valid_loader, crit, model)
best_score = 0
for i in range(120):
    top1_avg, loss = train(i, train_loader, model, opt, crit=crit, enorm=enorm, use_arc_metric=True)
    scheduler.step(top1_avg, i)
    top1_avg = validate(valid_loader, crit=crit, model=model)
    if i > 10:
        opt.swap_swa_sgd()
    if top1_avg > best_score:
        save_state(model.module, opt, top1_avg, loss,
                   name='/var/data/checkpoints_protein/arcface_effnet_b0_v3_epoch' + str(i) + '_.pth')
