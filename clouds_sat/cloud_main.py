import argparse

import segmentation_models_pytorch as smp
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback
from catalyst.dl.runner import SupervisedRunner
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from clouds_sat.cloud_data import ds_from_folds
from segmentation.custom_fpn import FPN
from segmentation.custom_unet import Unet

parser = argparse.ArgumentParser(description='Understanding cloud training')

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--seg-net', choices=['unet', 'fpn', 'ocunet'], default='fpn')
parser.add_argument('--loss', choices=['bce-dice', 'lovasz', 'weighted-bce', 'focal'], default='bce-dice')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--comment', type=str, default=None)
parser.add_argument('--swa', action='store_true')

parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/siim_acr_pneumo/train_png', required=False)
parser.add_argument('--folds-path', type=str, default='/home/lyan/Documents/kaggle/siim_acr_pnuemotorax/folds.csv',
                    required=False)
parser.add_argument('--mask-dir', type=str,
                    default='/var/ssd_1t/siim_acr_pneumo/masks_stage2/',
                    required=False)
parser.add_argument('--backbone', type=str, choices=['densenet121', 'densenet169', 'densenet201',
                                                     'densenet161', 'dpn68', 'dpn68b',
                                                     'dpn92', 'dpn98', 'dpn107', 'dpn131',
                                                     'inceptionresnetv2', 'resnet101', 'resnet152',
                                                     'se_resnet101', 'se_resnet152',
                                                     'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                                     'senet154', 'se_resnet50', 'resnet50', 'resnet34',
                                                     'efficientnet-b0', 'efficientnet-b1',
                                                     'efficientnet-b2', 'efficientnet-b3',
                                                     'efficientnet-b4', 'efficientnet-b5'],
                    default='se_resnext50_32x4d')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

ENCODER = args.backbone
if ENCODER == 'dpn92' or ENCODER == 'dpn68b':
    ENCODER_WEIGHTS = 'imagenet+5k'
else:
    ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'

ACTIVATION = 'sigmoid'

if args.seg_net == 'fpn':
    model = FPN(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=4, activation=ACTIVATION)
elif args.seg_net == 'unet':
    model = Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=4, activation=ACTIVATION)
elif args.seg_net == 'ocunet':
    model = Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=4, activation=ACTIVATION,
                 use_oc_module=True)
else:
    raise Exception('unsupported' + str(args.seg_net))

num_workers = 10
bs = args.batch_size

train_dataset, valid_dataset = ds_from_folds(path='/var/ssd_1t/cloud/', folds_path='train_folds.csv')

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 40
logdir = "/var/data/cloud/segmentation_f"+str(args.fold)+'_'+str(args.model)

optimizer = AdamW([
    {'params': model.decoder.parameters(), 'lr': args.lr},
    {'params': model.encoder.parameters(), 'lr': args.lr / 100.0},
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=5)
criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
runner = SupervisedRunner()

runner.train(
    fp16=args.fp16,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[DiceCallback(), EarlyStoppingCallback(patience=10, min_delta=0.001)],
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)
