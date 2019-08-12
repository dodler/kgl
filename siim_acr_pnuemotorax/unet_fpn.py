import sys

sys.path.append('/home/lyan/Documents/enorm/enorm')

import argparse
import datetime

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F

from siim_acr_pnuemotorax.segmentation.adamw import AdamW
from siim_acr_pnuemotorax.segmentation.albs import aug_light, aug_resize
from siim_acr_pnuemotorax.segmentation.custom_epoch import TrainEpoch, ValidEpoch
from siim_acr_pnuemotorax.segmentation.custom_fpn import FPN
from siim_acr_pnuemotorax.segmentation.custom_unet import Unet
from siim_acr_pnuemotorax.segmentation.focal_loss import FocalLoss2d
from siim_acr_pnuemotorax.segmentation.unet import lovasz_hinge
from siim_acr_pnuemotorax.siim_data import from_folds, SIIMDatasetSegmentation
from torchcontrib.optim import SWA

import os
import os.path as osp

from enorm import ENorm

NON_BEST_DONE_THRESH = 15

parser = argparse.ArgumentParser(description='SIIM ACR Pneumotorax unet training')

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--opt', default='Adam', choices=['SGD', 'Adam', 'Adamw'], type=str)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--opt-step-size', default=1, type=int)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--seg-net', choices=['unet', 'fpn', 'ocunet'], default='fpn')
parser.add_argument('--loss', choices=['bce-dice', 'lovasz', 'weighted-bce', 'focal'], default='bce-dice')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--comment', type=str, default=None)
parser.add_argument('--swa', action='store_true')

parser.add_argument('--image-dir',type=str, default='/var/ssd_1t/siim_acr_pneumo/train2017',required=False)
parser.add_argument('--folds-path',type=str, default='/home/lyan/Documents/kaggle/siim_acr_pnuemotorax/folds.csv',
                    required=False)
parser.add_argument('--mask-dir',type=str,
                    default='/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty/',
                    required=False)

parser.add_argument('--backbone-weights', type=str, default=None)
parser.add_argument('--backbone', type=str, choices=['densenet121', 'densenet169', 'densenet201',
                                                     'densenet161' 'dpn68', 'dpn68b',
                                                     'dpn92', 'dpn98', 'dpn107', 'dpn131',
                                                     'inceptionresnetv2', 'resnet101', 'resnet152',
                                                     'se_resnet101', 'se_resnet152',
                                                     'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                                     'senet154', 'se_resnet50', 'resnet50', 'resnet34',
                                                     'efficientnet-b0', 'efficientnet-b1',
                                                     'efficientnet-b2', 'efficientnet-b3',
                                                     'efficientnet-b4', 'efficientnet-b5'],
                    default='se_resnext50_32x4d')
parser.add_argument('--enorm', action='store_true')
args = parser.parse_args()

ENCODER = args.backbone
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = ['pneumo']
ACTIVATION = 'sigmoid'

if args.seg_net == 'fpn':
    model = FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION)
elif args.seg_net == 'unet':
    model = Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION)
elif args.seg_net == 'ocunet':
    model = Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION,
                 use_oc_module=True)
else:
    raise Exception('unsupported' + str(args.seg_net))

if args.backbone_weights is not None:
    pretrained_dict = torch.load(args.backbone_weights, map_location='cpu')['model']

    model_dict = model.encoder.state_dict()
    pretrained_dict = {k.replace('feature_extr.', ''): v for k, v in pretrained_dict.items()
                       if k.replace('feature_extr.', '') in model_dict}
    model_dict.update(pretrained_dict)
    # model_dict['last_linear.bias'] = None
    # model_dict['last_linear.weight'] = None
    model.encoder.load_state_dict(model_dict)

model.to(0)


def weighted_bce(logit_pixel, gt):
    logit = logit_pixel.view(-1)
    truth = gt.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()

    return loss


def lovasz_and_dice(pred, gt):
    return lovasz_hinge(pred, gt)


if args.loss == 'bce-dice':
    loss = smp.utils.losses.BCEDiceLoss(eps=1.)
elif args.loss == 'lovasz':
    loss = lovasz_hinge
elif args.loss == 'weighted-bce':
    loss = weighted_bce
elif args.loss == 'focal':
    loss = FocalLoss2d()  # fixme, not working still
else:
    raise Exception('unsupported loss', args.loss)

metrics = [
    smp.utils.metrics.IoUMetric(eps=1.),
    smp.utils.metrics.FscoreMetric(eps=1.),
]

params = [
    {'params': model.decoder.parameters(), 'lr': args.lr},
    {'params': model.encoder.parameters(), 'lr': args.lr / 100.0},
]

if args.opt == 'Adam':
    optimizer = torch.optim.Adam(params, weigth_decay=5e-4)
elif args.opt == 'SGD':
    optimizer = torch.optim.SGD(params, momentum=0.9)
elif args.opt == 'Adamw':
    optimizer = AdamW(params, weight_decay=5e-4)

if args.resume is not None:
    state = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

if args.swa:
    opt = SWA(optimizer)

if args.enorm:
    enorm = ENorm(model.encoder.named_parameters(), optimizer, model_type='conv', c=1)  # fixme, pull to arguments
else:
    enorm = None

model = torch.nn.DataParallel(model)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples

comment = ''
if args.comment is not None:
    comment = args.comment

experiment_name = args.backbone + '_' \
                  + args.seg_net + '_' \
                  + str(args.opt) + '_' \
                  + str(args.batch_size) + '_' \
                  + str(args.loss) + '_' \
                  + 'fold' + str(args.fold) + '_' \
                  + comment + '_'

if args.opt_step_size > 1:
    experiment_name += '_step_size_' + str(args.opt_step_size) + '_'

if args.swa:
    experiment_name += '_swa_'

if args.enorm:
    experiment_name += '_enorm_'

if args.backbone_weights is not None:
    experiment_name += '_backbone_weights_' + args.backbone_weights.replace('/', '_') + '_'

experiment_name += datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

experiment_name = experiment_name.replace('__', '_')

train_epoch = TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    opt_step_size=args.opt_step_size,
    verbose=True,
    experiment_name=experiment_name,
    enorm=enorm
)

valid_epoch = ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
    experiment_name=experiment_name,
)

# train model for 40 epochs

if args.fold == -1:
    train_dataset = SIIMDatasetSegmentation(image_dir=args.image_dir,
                                            mask_dir=args.mask_dir,
                                            aug=aug_light,
                                            # preprocessing_fn=get_preprocessing(preprocessing_fn)
                                            )
    valid_dataset = SIIMDatasetSegmentation(image_dir=args.image_dir,
                                            mask_dir=args.mask_dir,
                                            aug=aug_resize,
                                            # preprocessing_fn=get_preprocessing(preprocessing_fn)
                                            )
else:
    train_dataset, valid_dataset = from_folds(image_dir=args.image_dir,
                                              mask_dir=args.mask_dir,
                                              folds_path=args.folds_path,
                                              aug_trn=aug_light,
                                              aug_val=aug_resize,
                                              fold=args.fold)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          num_workers=8)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                          num_workers=8)

max_score = 0

lr = args.lr


def upd_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


upd_lr(optimizer, args.lr)


def save_state(model, epoch, opt, lr, score, val_loss, train_loss):
    if not osp.exists('/var/data/checkpoints/' + experiment_name):
        os.mkdir('/var/data/checkpoints/' + experiment_name)

    state = {'net': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'lr': lr, 'score': score,
             'val_loss': val_loss, 'train_loss': train_loss}
    filename = '/var/data/checkpoints/' + experiment_name + '/epoch_' + str(epoch) + '.pth'
    torch.save(state, filename)
    print('dumped to ', filename)


scheduler = ReduceLROnPlateau(optimizer, verbose=True, mode='max', patience=7)

best_metric = -1

non_best_epochs_done = 0

for i in range(0, args.epochs):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    fscore = valid_logs['f-score']
    if fscore > best_metric:
        best_metric = fscore
        save_state(model.module, epoch=i, opt=optimizer, lr=lr, score=fscore, val_loss=-1, train_loss=-1)
        non_best_epochs_done = 0
    else:
        non_best_epochs_done += 1

    if non_best_epochs_done > NON_BEST_DONE_THRESH:
        print('doing early stopping')
        break

    scheduler.step(fscore, i)

if args.swa:
    opt.swap_swa_sgd()
    valid_logs = valid_epoch.run(valid_loader)
    save_state(model.module, epoch=-1, opt=optimizer, lr=lr, score=fscore, val_loss=-1, train_loss=-1)
