import argparse
import datetime
import os
import os.path as osp
import sys

from torch.nn import Conv2d
import pandas as pd
import numpy as np

from severstal.sev_data import SevPretrain
from severstal.utils import train_classif

SEED = 42
from catalyst.utils import set_global_seed, prepare_cudnn

set_global_seed(SEED)
prepare_cudnn(deterministic=True)

import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

sys.path.append('/home/lyan/Documents/Synchronized-BatchNorm-PyTorch')
from sync_batchnorm import convert_model

NON_BEST_DONE_THRESH = 15

parser = argparse.ArgumentParser(description='Severstal segmentation train')

parser.add_argument('--backbone',required=True)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/severstal/train/', required=False)
args = parser.parse_args()

model = EfficientNet.from_pretrained(args.backbone)
model._fc = torch.nn.Linear(1408, 20)
model._conv_stem = Conv2d(3, 32, kernel_size=3, stride=2, bias=False)

torch.nn.init.xavier_normal_(model._fc.weight)
torch.nn.init.xavier_normal_(model._conv_stem.weight)

for p in model.parameters():
    p.requires_grad = True

model.cuda()

experiment_name = 'pretrain_' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
experiment_name=experiment_name.replace('/','_')

pretrain = pd.read_csv('/home/lyan/Documents/kaggle/severstal/pretrain.csv')['ImageId_ClassId'].values.tolist()
pretrain=np.array([k.split('_')[0] for k in pretrain])
train_dataset = SevPretrain(img_ids=pretrain, image_dir=args.image_dir, aug=None)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          num_workers=args.num_workers)

lr = 0.1
logdir = "/var/data/logdir/"
num_epochs = 42

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
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
    torch.save(model.state_dict(), filename)
    print('dumped to ', filename)


best_score = 0
for i in range(args.epochs):
    top1, loss = train_classif(i, train_loader, model, optimizer, criterion, experiment_name)
save_state(model, i, optimizer, -1, -1, -1, loss)
