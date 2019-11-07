import argparse
import os
import os.path as osp

import numpy as np

from rsna_hemorr.hem_data import hem_png_from_folds, hem_dcm_from_folds
from rsna_hemorr.hem_utils import get_model
from rsna_hemorr.losses import FocalLoss

dir_csv = '/var/ssd_1t/rsna_intra_hemorr/'
dir_train_img = '/var/ssd_1t/rsna_intra_hemorr/stage_1_train_png_224x/'
dir_test_img = '/var/ssd_1t/rsna_intra_hemorr/stage_1_test_png_224x/'

n_classes = 6
n_epochs = 4
batch_size = 128

import sys

# sys.path.append('/home/lyan/Documents/Synchronized-BatchNorm-PyTorch')
sys.path.append('/home/lyan/Synchronized-BatchNorm-PyTorch')
from sync_batchnorm import convert_model

# Libraries

import torch

device = torch.device("cuda:0")

import torch.optim as optim
from apex import amp
from torch.utils.data import Dataset
from tqdm import *

parser = argparse.ArgumentParser(description='rsna hemorr train')

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
parser.add_argument('--folds-path', type=str, default='stage_1_train_folds.csv',
                    required=False)
parser.add_argument('--raw', action='store_true')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--image-path',type=str,required=False, default='/var/ssd_1t/rsna_intra_hemorr/stage_1_train_png_224x/')
args = parser.parse_args()

print('train with args', args)

# if parser.raw:
#     train_dataset, valid_dataset = hem_dcm_from_folds(fold=args.fold)
# else:
train_dataset, valid_dataset = hem_png_from_folds(image_path=args.image_path, fold=args.fold)
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=8)
data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=8)

model = get_model(args.model)


def save_state(model, epoch):
    if MULTI_DEVICE:
        ckpt = model.module.state_dict()
    else:
        ckpt = model.state_dict()

    name = osp.join('.rsna_ckpts/', args.model + '_f_' + str(args.fold) ,'epoch_'+str(epoch)+ '.pth')
    if not osp.exists(osp.join('.rsna_ckpts/', args.model + '_f_' + str(args.fold))):
        os.mkdir(osp.join('./rsna_ckpts/', args.model + '_f_' + str(args.fold)))

    torch.save({
        'model': ckpt,
    }, name)
    print('saved to', name)


if args.resume is not None:
    raise Exception('not ready yet')

MULTI_DEVICE = torch.cuda.device_count() > 1

if MULTI_DEVICE:
    model = convert_model(model)

model.cuda()

# criterion = torch.nn.BCEWithLogitsLoss()
criterion = FocalLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if MULTI_DEVICE:
    model = torch.nn.DataParallel(model)

best_loss = 1e09
for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()
    tr_loss = 0

    tk0 = tqdm(data_loader_train, desc="Iteration")

    for step, batch in enumerate(tk0):
        # inputs = batch["image"]
        inputs = batch[0]
        # labels = batch["labels"]
        labels = batch[1]

        inputs = inputs.cuda().float()
        labels = labels.cuda().float()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))

    model.eval()
    tk0 = tqdm(data_loader_valid, desc="Iteration")
    losses = []
    for step, batch in enumerate(tk0):
        # inputs = batch["image"]
        inputs = batch[0]
        # labels = batch["labels"]
        labels = batch[1]

        inputs = inputs.cuda().float()
        labels = labels.cuda().float()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

    val_loss = np.array(losses).mean()
    # if val_loss < best_loss:
    save_state(model, epoch)
    print('valid loss', val_loss)
