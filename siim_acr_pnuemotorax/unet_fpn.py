import argparse
import datetime
import uuid

import segmentation_models_pytorch as smp
import torch
from albumentations import Compose, Lambda
from torch.utils.data import DataLoader

from siim_acr_pnuemotorax.segmentation.albs import aug_geom_color, aug_resize, aug_light
from siim_acr_pnuemotorax.segmentation.custom_epoch import TrainEpoch, ValidEpoch
from siim_acr_pnuemotorax.segmentation.unet import symmetric_lovasz, lovasz_hinge
from siim_acr_pnuemotorax.siim_data import SIIMDatasetSegmentation

parser = argparse.ArgumentParser(description='SIIM ACR Pneumotorax unet training')

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--opt', default='Adam', choices=['SGD', 'Adam'], type=str)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--opt-step-size', default=1, type=int)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--seg-net', choices=['unet', 'fpn'], default='fpn')
parser.add_argument('--backbone', type=str, choices=['densenet121', 'densenet169', 'densenet201', 'densenet161'
                                                                                                  'dpn68', 'dpn68b',
                                                     'dpn92', 'dpn98', 'dpn107', 'dpn131',
                                                     'inceptionresnetv2', 'resnet101', 'resnet152',
                                                     'se_resnet101', 'se_resnet152',
                                                     'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                                     'senet154', 'se_resnet50', 'resnet50', 'resnet34'],
                    default='se_resnext50_32x4d')
args = parser.parse_args()

ENCODER = args.backbone
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

CLASSES = ['pneumo']
ACTIVATION = 'sigmoid'

if args.seg_net == 'fpn':
    model = smp.FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION)
elif args.seg_net == 'unet':
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, activation=ACTIVATION)
else:
    raise Exception('unsupported' + str(args.seg_net))

model.to(0)

loss = smp.utils.losses.BCEDiceLoss(eps=1.)


def lovasz_and_dice(pred, gt):
    return lovasz_hinge(pred, gt)
    # return loss(gt, pred)


metrics = [
    smp.utils.metrics.IoUMetric(eps=1.),
    smp.utils.metrics.FscoreMetric(eps=1.),
]

if args.opt == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.opt == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)

if args.resume is not None:
    state = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

model = torch.nn.DataParallel(model)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples

experiment_name = args.backbone + '_' + args.seg_net + '_' + str(args.opt) + '_' + str(args.batch_size) + '_' + datetime.datetime.now().strftime(
    "%Y-%m-%d_%H_%M_%S")


train_epoch = TrainEpoch(
    model,
    loss=lovasz_and_dice,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    opt_step_size=args.opt_step_size,
    verbose=True,
    experiment_name=experiment_name,
)

valid_epoch = ValidEpoch(
    model,
    loss=lovasz_and_dice,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
    experiment_name=experiment_name,
)

# train model for 40 epochs


train_dataset = SIIMDatasetSegmentation(image_dir='/var/ssd_1t/siim_acr_pneumo/train2017',
                                        mask_dir='/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty/',
                                        aug=aug_light,
                                        # preprocessing_fn=get_preprocessing(preprocessing_fn)
                                        )
valid_dataset = SIIMDatasetSegmentation(image_dir='/var/ssd_1t/siim_acr_pneumo/val2017',
                                        mask_dir='/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty/',
                                        aug=None,
                                        # preprocessing_fn=get_preprocessing(preprocessing_fn)
                                        )

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
    state = {'net': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'lr': lr, 'score': score,
             'val_loss': val_loss, 'train_loss': train_loss}
    filename = '/var/data/checkpoints/' + args.backbone + '_' + args.seg_net + '_' + str(
        epoch) + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.pth'
    torch.save(state, filename)
    print('dumped to ', filename)


from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, verbose=True)

try:
    for i in range(0, 120):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        # if max_score < valid_logs['iou']:
        #     max_score = valid_logs['iou']
        save_state(model.module, epoch=i, opt=optimizer, lr=lr, score=max_score, val_loss=-1, train_loss=-1)
        # print('Model saved!')

        fscore = train_logs['f-score']
        scheduler.step(fscore, i)

        # if i == 40 or i == 80:
        #     lr /= 10.0
        #     train_epoch.opt_step_size /= 2
        #     optimizer.param_groups[0]['lr'] = lr
        #     print('Decrease decoder learning rate to !')
except KeyboardInterrupt:
    save_state(model=model.module, epoch=i, opt=optimizer, lr=lr, score=-1, val_loss=-1, train_loss=-1)

save_state(model=model.module, epoch=i, opt=optimizer, lr=lr, score=-1, val_loss=-1, train_loss=-1)
