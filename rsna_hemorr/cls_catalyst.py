import argparse
import datetime
from collections import OrderedDict

import numpy as np
from catalyst.dl import MultiMetricCallback
from catalyst.dl.callbacks import EarlyStoppingCallback
from catalyst.dl.runner import SupervisedRunner
from sklearn.metrics import log_loss

from rsna_hemorr.hem_data import hem_png_from_folds, hem_dcm_from_folds
from rsna_hemorr.hem_utils import get_model

dir_csv = '/var/ssd_1t/rsna_intra_hemorr/'
dir_train_img = '/var/ssd_1t/rsna_intra_hemorr/stage_1_train_png_224x/'
dir_test_img = '/var/ssd_1t/rsna_intra_hemorr/stage_1_test_png_224x/'

n_classes = 6
n_epochs = 4
batch_size = 128

import sys

sys.path.append('/home/lyan/Documents/Synchronized-BatchNorm-PyTorch')
from sync_batchnorm import convert_model

# Libraries

import torch

device = torch.device("cuda:0")

import torch.optim as optim
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description='rsna hemorr train with catalyst')

parser.add_argument('--lr',
                    default=2e-5,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=14)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--comment', type=str, default=None)
# parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/severstal/train/', required=False)
parser.add_argument('--folds-path', type=str, default='stage_1_train_folds.csv',
                    required=False)
parser.add_argument('--raw', action='store_true')
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

print('train with args', args)

if args.raw:
    train_dataset, valid_dataset = hem_dcm_from_folds(fold=args.fold)
else:
    train_dataset, valid_dataset = hem_png_from_folds(fold=args.fold)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=14)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=14)

model = get_model(args.model, raw=args.raw)

if args.resume is not None:
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model_state_dict'])

MULTI_DEVICE = torch.cuda.device_count() > 1

if MULTI_DEVICE:
    model = convert_model(model)

model.cuda()

plist = [{'params': model.parameters(), 'lr': args.lr}]
optimizer = optim.Adam(plist, lr=args.lr)

experiment_name = 'cls_' + args.model + '_' + '_f_'+str(args.fold)+'_' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
experiment_name = experiment_name.replace('/', '_')

lr = args.lr


loss_weights = torch.tensor([2.0, 1, 1, 1, 1, 1]).cuda()*0.14*2
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, mode='min')
criterion = torch.nn.BCEWithLogitsLoss(weight=loss_weights).float()
loaders = OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader

num_epochs = args.epochs
logdir = "/var/data/logs/" + experiment_name
runner = SupervisedRunner()


def calc_metric(pred, gt, *args, **kwargs):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy().astype(np.uint8)
    try:
        return [log_loss(gt.reshape(-1), pred.reshape(-1))]
    except:
        return [0]


runner.train(
    fp16=True,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    scheduler=scheduler,
    num_epochs=num_epochs,
    callbacks=[
        MultiMetricCallback(metric_fn=calc_metric, prefix='logloss',
                            input_key="targets",
                            output_key="logits",
                            list_args=['_']),
        EarlyStoppingCallback(patience=5, min_delta=0.01)
    ],
    verbose=True
)
