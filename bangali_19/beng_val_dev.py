import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm import *
from sklearn.metrics import accuracy_score, recall_score

import pandas as pd
from bangali_19.beng_augs import valid_aug_v0
from bangali_19.beng_data import BengaliDataset
from bangali_19.beng_densenet import BengDensenet
from bangali_19.beng_eff_net import BengEffNetClassifier
import numpy as np

from bangali_19.beng_resnets import BengResnet
from bangali_19.beng_utils import get_dict_value_or_default
from bangali_19.configs import get_config

parser = argparse.ArgumentParser(description='Understanding cloud training')

parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--config', type=str, required=True, default=None)
parser.add_argument('--ckpt', type=str, required=True, default=None)
args = parser.parse_args()

# /var/data/bengali0_model_efficientnet-b0_comment_bn_and_pretrain/checkpoints//train.30.pth
ckpt = torch.load(args.ckpt)['model_state_dict']

config = get_config(name=args.config)

iafoss_head = get_dict_value_or_default(config, key='isfoss_head', default_value=False)
isfoss_norm = get_dict_value_or_default(config, key='isfoss_norm', default_value=False)

dropout = get_dict_value_or_default(config, key='dropout', default_value=0.2)
head = get_dict_value_or_default(config, key='head', default_value='V0')

if config['arch'] == 'multi-head':
    if 'efficientnet' in config['backbone']:
        model = BengEffNetClassifier(
            name=config['backbone'],
            pretrained=config['pretrained'],
            input_bn=config['in-bn'],
            dropout=dropout,
            head=head,
        )
    elif 'resnet' in config['backbone'] or 'resnext' in config['backbone']:
        model = BengResnet(
            name=config['backbone'],
            pretrained=config['pretrained'],
            input_bn=config['in-bn'],
            dropout=dropout,
            isfoss_head=iafoss_head,
            head=head,
        )
    elif 'densenet' in config['backbone']:
        model = BengDensenet(
            name=config['backbone'],
            input_bn=config['in-bn'],
            dropout=dropout,
            isfoss_head=iafoss_head,
            head=head,
        )
    else:
        raise Exception('backbone ' + config['backbone'] + ' is not supported')
else:
    raise Exception(config['arch'] + ' is not supported')


model.eval()
model.to(0)
model.load_state_dict(ckpt)

img_path = '/var/ssd_1t/kaggle_bengali/jpeg_crop/'
dev_df = pd.read_csv('/home/lyan/Documents/kaggle/bangali_19/dev.csv')
dev_ids = dev_df.values
dev_dataset = BengaliDataset(path=img_path, values=dev_ids, aug=valid_aug_v0)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

h1_preds = []
h2_preds = []
h3_preds = []

h1_gt = dev_dataset.values[:, 1].astype(np.int32)
h2_gt = dev_dataset.values[:, 2].astype(np.int32)
h3_gt = dev_dataset.values[:, 3].astype(np.int32)

st = time.time()

for batch in tqdm(dev_loader):
    img = batch['features'].to(0)

    with torch.no_grad():
        h1, h2, h3 = model(img)
        h1 = torch.argmax(h1, dim=1).detach().cpu().numpy()
        h2 = torch.argmax(h2, dim=1).detach().cpu().numpy()
        h3 = torch.argmax(h3, dim=1).detach().cpu().numpy()

    h1_preds.append(h1)
    h2_preds.append(h2)
    h3_preds.append(h3)

h1_preds = np.concatenate(h1_preds)
h2_preds = np.concatenate(h2_preds)
h3_preds = np.concatenate(h3_preds)

print(h1_preds.shape, h1_gt.shape)

scores = [recall_score(y_true=h1_gt, y_pred=h1_preds, average='macro'),
          recall_score(y_true=h2_gt, y_pred=h2_preds, average='macro'),
          recall_score(y_true=h3_gt, y_pred=h3_preds, average='macro')]

scores = np.array(scores)

print('avg recall', np.average(scores, weights=[2, 1, 1]))
print('time elapsed ', time.time() - st)
