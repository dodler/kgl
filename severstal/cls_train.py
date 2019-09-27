import argparse
import datetime
import os
import os.path as osp
import sys

from albumentations import Resize
from torch.nn import Conv2d

from segmentation.adamw import AdamW
from severstal.utils import train_classif, validate_classif

SEED = 42
from catalyst.utils import set_global_seed, prepare_cudnn

set_global_seed(SEED)
prepare_cudnn(deterministic=True)

import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

from config.seg_config import from_json
from severstal.augs_severstal import aug_light_cls
from severstal.sev_utils import cls_from_folds

sys.path.append('/home/lyan/Documents/Synchronized-BatchNorm-PyTorch')
from sync_batchnorm import convert_model

NON_BEST_DONE_THRESH = 15

parser = argparse.ArgumentParser(description='Severstal segmentation train')

parser.add_argument('--lr',
                    default=0.1,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--opt-step-size', default=1, type=int)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--comment', type=str, default=None)

parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/severstal/train/', required=False)
parser.add_argument('--folds-path', type=str, default='/home/lyan/Documents/kaggle/severstal/train.csv',
                    required=False)
parser.add_argument('--mask-dir', type=str,
                    default='/var/ssd_1t/severstal/train/',
                    required=False)
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

conf = from_json(args.config)

model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(1280, 1)

torch.nn.init.xavier_normal_(model._fc.weight)
torch.nn.init.xavier_normal_(model._conv_stem.weight)

for p in model.parameters():
    p.requires_grad = True

if torch.cuda.device_count() > 1:
    model = convert_model(model)

if torch.cuda.is_available():
    model.cuda()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

experiment_name = 'cls_' + args.config + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
experiment_name=experiment_name.replace('/','_')

train_dataset, valid_dataset = cls_from_folds(image_dir=args.image_dir,
                                              folds_path=args.folds_path,
                                              aug_trn=aug_light_cls,
                                              aug_val=Resize(128,800),
                                              fold=args.fold)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                          num_workers=args.num_workers)
lr = args.lr
logdir = "/var/data/logdir/"
num_epochs = 42

criterion = torch.nn.BCEWithLogitsLoss()
opt=getattr(conf, 'opt', None)

if opt is None or opt == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif opt == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
elif opt == 'Adamw':
    optimizer=AdamW(model.parameters(), lr=args.lr)
else:
    raise Exception('optimizer '+opt+' is not supported')


if args.resume is not None:
    state = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)


def save_state(model, epoch, opt, lr, score, val_loss, train_loss):
    if not osp.exists('/var/data/checkpoints/' + experiment_name):
        os.mkdir('/var/data/checkpoints/' + experiment_name)

    if torch.cuda.device_count() > 2:
        state = {'net': model.module.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'lr': lr, 'score': score,
                 'val_loss': val_loss, 'train_loss': train_loss}
    else:
        state = {'net': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch, 'lr': lr, 'score': score,
                 'val_loss': val_loss, 'train_loss': train_loss}

    filename = '/var/data/checkpoints/' + experiment_name + '/epoch_' + str(epoch) + '.pth'
    torch.save(state, filename)
    print('dumped to ', filename)


best_score = 0
for i in range(args.epochs):
    top1, loss = train_classif(i, train_loader, model, optimizer, criterion, experiment_name)
    top1_val = validate_classif(valid_loader, criterion, model, experiment_name)
    print(top1_val)
    if top1_val > best_score:
        best_score = top1_val
        save_state(model, i, optimizer, -1, top1_val, -1, loss)
