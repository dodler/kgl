import argparse
import datetime
import sys
from collections import OrderedDict

from catalyst.dl import MultiMetricCallback
from catalyst.dl.runner import SupervisedRunner
from albumentations import Resize
from segmentation.adamw import AdamW
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback

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


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


model = EfficientNet.from_pretrained('efficientnet-b2')
model._fc = torch.nn.Sequential(torch.nn.Linear(1408, 4), Flatten())
# model._fc = torch.nn.Linear(1280, 1)

for p in model.parameters():
    p.requires_grad = True

if torch.cuda.device_count() > 1:
    model = convert_model(model)

if torch.cuda.is_available():
    model.cuda()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

experiment_name = 'cls_' + args.config + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
experiment_name = experiment_name.replace('/', '_')

train_dataset, valid_dataset = cls_from_folds(image_dir=args.image_dir,
                                              folds_path=args.folds_path,
                                              aug_trn=aug_light_cls,
                                              aug_val=Resize(256, 1600),
                                              fold=args.fold)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size,
                          num_workers=args.num_workers)
lr = args.lr
logdir = "/var/data/logdir/"
num_epochs = 42

opt = getattr(conf, 'opt', None)

if opt is None or opt == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif opt == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
elif opt == 'Adamw':
    optimizer = AdamW(model.parameters(), lr=args.lr)
else:
    raise Exception('optimizer ' + opt + ' is not supported')

if args.resume is not None:
    state = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

lr = 0.1
criterion = torch.nn.BCEWithLogitsLoss().float()
loaders = OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader

num_epochs = args.epochs
logdir = "/var/data/logs/" + experiment_name
runner = SupervisedRunner()

import numpy as np
from sklearn.metrics import roc_auc_score


def calc_roc_auc(pred, gt, *args, **kwargs):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy().astype(np.uint8)

    pred = np.concatenate([pred.reshape(-1), np.array([0, 0])])
    gt = np.concatenate([gt.reshape(-1), np.array([1, 0])])

    return [roc_auc_score(gt.reshape(-1), pred.reshape(-1))]


runner.train(
    model=model,
    scheduler=scheduler,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    callbacks=[
        MultiMetricCallback(metric_fn=calc_roc_auc, prefix='rocauc',
                            input_key="targets",
                            output_key="logits",
                            list_args=['_']),
        EarlyStoppingCallback(patience=10, min_delta=0.01)
    ],
    verbose=True
)
