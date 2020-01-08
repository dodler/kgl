import torch
from torch.utils.data import DataLoader
from tqdm import *
from sklearn.metrics import accuracy_score

import pandas as pd
from bangali_19.beng_augs import valid_aug_v0
from bangali_19.beng_data import BengaliDataset
from bangali_19.beng_eff_net import BengEffNetClassifier
import numpy as np

ckpt = torch.load('/var/data/bengali0_model_efficientnet-b0_comment_bn_and_pretrain/checkpoints//train.30.pth')[
    'model_state_dict']
model = BengEffNetClassifier()
model.eval()
model.to(0)
model.load_state_dict(ckpt)

img_path = '/var/ssd_1t/kaggle_bengali/jpeg_crop/'
dev_df = pd.read_csv('/home/lyan/Documents/kaggle/bangali_19/dev.csv')
dev_ids = dev_df.values
dev_dataset = BengaliDataset(path=img_path, values=dev_ids, aug=valid_aug_v0)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, num_workers=10)

h1_preds = []
h2_preds = []
h3_preds = []

h1_gt = dev_dataset.values[:, 1].astype(np.int32)
h2_gt = dev_dataset.values[:, 2].astype(np.int32)
h3_gt = dev_dataset.values[:, 3].astype(np.int32)

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

print(accuracy_score(y_true=h1_gt, y_pred=h1_preds))
print(accuracy_score(y_true=h2_gt, y_pred=h2_preds))
print(accuracy_score(y_true=h3_gt, y_pred=h3_preds))
