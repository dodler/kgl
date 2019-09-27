import sys

from config.seg_config import from_json
from segmentation.adamw import AdamW
from segmentation.custom_fpn import FPN
from segmentation.custom_unet import Unet
from segmentation.losses import weighted_bce
from segmentation.segmentation.losses import lovasz_hinge
from severstal.augs_severstal import aug_light, aug_resize
from severstal.sev_utils import load_pretrained_weights, seg_from_folds
from siim_acr_pnuemotorax.prediction_utils import DiceMetric
import argparse
import datetime

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

from segmentation.custom_epoch import TrainEpoch, ValidEpoch
from torchcontrib.optim import SWA

import os
import os.path as osp

from apex import amp

sys.path.append('/home/lyan/Documents/Synchronized-BatchNorm-PyTorch')
from sync_batchnorm import convert_model
from pytorch_toolbelt.losses.focal import *

NON_BEST_DONE_THRESH = 15

parser = argparse.ArgumentParser(description='Severstal segmentation train')

parser.add_argument('--lr',
                    default=1e-2,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--opt-step-size', default=1, type=int)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--comment', type=str, default=None)

parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/severstal/img_crops/', required=False)
parser.add_argument('--folds-path', type=str, default='/home/lyan/Documents/kaggle/severstal/crop_folds.csv',
                    required=False)
parser.add_argument('--mask-dir', type=str,
                    default='/var/ssd_1t/severstal/mask_crops/',
                    required=False)
parser.add_argument('--opt-level', type=str, default=None, required=False, choices=['O1', 'O2'])
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

conf = from_json(args.config)

ENCODER = conf.backbone
if ENCODER == 'dpn92' or ENCODER == 'dpn68b':
    ENCODER_WEIGHTS = 'imagenet+5k'
else:
    ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = ['pneumo']
ACTIVATION = 'sigmoid'

if conf.seg_net == 'fpn':
    model = FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION)
elif conf.seg_net == 'unet':
    model = Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION)
elif conf.seg_net == 'ocunet':
    model = Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION,
                 use_oc_module=True)
else:
    raise Exception('unsupported' + str(args.seg_net))

if conf.backbone_weights is not None:
    load_pretrained_weights(conf.backbone_weights, model)

if torch.cuda.device_count() > 1:
    model = convert_model(model)

if torch.cuda.is_available():
    model.cuda()

if conf.loss == 'bce-dice':
    loss = smp.utils.losses.BCEDiceLoss(eps=1.)
elif conf.loss == 'lovasz':
    loss = lovasz_hinge
elif conf.loss == 'weighted-bce':
    loss = weighted_bce
elif conf.loss == 'focal':
    loss = BinaryFocalLoss()
    loss.__name__ = 'bin_focal'
else:
    raise Exception('unsupported loss', args.loss)

metrics = [
    smp.utils.metrics.IoUMetric(eps=1.),
    DiceMetric()
]

params = [
    {'params': model.decoder.parameters(), 'lr': args.lr},
    {'params': model.encoder.parameters(), 'lr': args.lr / 100.0},
]

if conf.opt == 'Adam':
    optimizer = torch.optim.Adam(params, weigth_decay=5e-4)
elif conf.opt == 'SGD':
    optimizer = torch.optim.SGD(params, momentum=0.9,weight_decay=5e-4)
elif conf.opt == 'Adamw':
    optimizer = AdamW(params, weight_decay=5e-4)

if args.resume is not None:
    state = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state['net'])
    # optimizer.load_state_dict(state['opt'])

if conf.swa:
    optimizer = SWA(optimizer, swa_start=2, swa_freq=5, swa_lr=args.lr / 2)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

if args.opt_level is not None:
    print('using opt level', args.opt_level)
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

experiment_name = args.config.replace('/','_')
experiment_name += '_fold_'+str(args.fold)
experiment_name += '_bs_'+str(args.batch_size)
experiment_name += '_optl_'+str(args.opt_level)
experiment_name += '_'+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

train_epoch = TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    opt_step_size=args.opt_step_size,
    verbose=True,
    experiment_name=experiment_name,
)

valid_epoch = ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
    experiment_name=experiment_name,
)

aug_trn = None
aug_val = None

from albumentations.core.serialization import save, load

if getattr(conf, "augs_trn", None) is not None:
    aug_trn = load(conf.augs_trn)
else:
    aug_trn = aug_light

if getattr(conf, "augs_val", None) is not None:
    aug_val = load(conf.augs_val)
else:
    aug_val = aug_resize

train_dataset, valid_dataset = seg_from_folds(image_dir=args.image_dir,
                                              mask_dir=args.mask_dir,
                                              folds_path=args.folds_path,
                                              aug_trn=aug_trn,
                                              aug_val=aug_val,
                                              fold=args.fold)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                          num_workers=args.num_workers)
max_score = 0

lr = args.lr


def upd_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


upd_lr(optimizer, args.lr)


def save_state(model, epoch, opt, lr, score, val_loss, train_loss, args):
    if not osp.exists('/var/data/checkpoints/' + experiment_name):
        os.mkdir('/var/data/checkpoints/' + experiment_name)

    if torch.cuda.device_count() > 2:
        state = {'net': model.module.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'lr': lr, 'score': score,
                 'val_loss': val_loss, 'train_loss': train_loss}
    else:
        state = {'net': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'lr': lr, 'score': score,
                 'val_loss': val_loss, 'train_loss': train_loss}

    state.update(vars(args))

    filename = '/var/data/checkpoints/' + experiment_name + '/epoch_' + str(epoch) + '.pth'
    torch.save(state, filename)
    print('dumped to ', filename)


if getattr(conf, 'sched', None) == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
else:
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, mode='max', patience=7)
if getattr(conf, 'warmup', None) == 'True':
    from warmup_scheduler import GradualWarmupScheduler
    scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=scheduler)

best_metric = -1
non_best_epochs_done = 0

for i in range(0, args.epochs):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    fscore = valid_logs['dice']
    if fscore > best_metric:
        best_metric = fscore
        save_state(model, epoch=i, opt=optimizer, lr=lr, score=fscore, val_loss=-1, train_loss=-1, args=args)
        non_best_epochs_done = 0
    else:
        non_best_epochs_done += 1

    if non_best_epochs_done > NON_BEST_DONE_THRESH:
        print('doing early stopping')
        break

    scheduler.step(fscore, i)

if conf.swa:
    optimizer.swap_swa_sgd()
    valid_logs = valid_epoch.run(valid_loader)
    save_state(model, epoch=-1, opt=optimizer, lr=lr, score=fscore, val_loss=-1, train_loss=-1, args=args)
