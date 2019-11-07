import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import *

from rsna_hemorr.hem_augs import get_png_tta, get_raw_tta
from rsna_hemorr.hem_data import IntracranialDatasetRaw, IntracranialDataset
from rsna_hemorr.hem_utils import get_model
from rsna_hemorr.losses import FocalLoss

dir_csv = '/var/ssd_1t/rsna_intra_hemorr/'

n_classes = 6
n_epochs = 5
batch_size = 64

test = pd.read_csv('test.csv')

parser = argparse.ArgumentParser(description='rsna hemorr train')

parser.add_argument('--image-path', type=str, default=None)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--raw', action='store_true')
args = parser.parse_args()


def predict_tfm(tfm):
    if args.raw:
        ds = IntracranialDatasetRaw(test, path=args.image_path, transform=tfm, labels=False)
    else:
        ds = IntracranialDataset(test, path=args.image_path, transform=tfm, labels=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    result = np.zeros((len(ds) * n_classes, 1))

    for i, x_batch in enumerate(tqdm(dl)):
        x_batch = x_batch.to(device, dtype=torch.float)

        with torch.no_grad():
            pred = model(x_batch)

            result[(i * batch_size * n_classes):((i + 1) * batch_size * n_classes)] = torch.sigmoid(
                pred).detach().cpu().reshape((len(x_batch) * n_classes, 1))

    return result


device = torch.device("cuda:0")

model = get_model(args.model, raw=args.raw)
model.to(device)

criterion = FocalLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


print('loading state dict', args.resume)

# if 'model' in
checkpoint = torch.load(args.resume)
if 'model' in checkpoint:
    model_state_dict = checkpoint['model']
else:
    model_state_dict = checkpoint['model_state_dict']

model.load_state_dict(model_state_dict)

for param in model.parameters():
    param.requires_grad = False

model.eval()


if args.raw:
    tta_transforms = get_raw_tta()
else:
    tta_transforms = get_png_tta()

test_pred = predict_tfm(tta_transforms[0])
for tta_tfm in tta_transforms[1:]:
    test_pred += predict_tfm(tta_tfm)

test_pred /= float(len(tta_transforms))


submission = pd.read_csv('stage_1_sample_submission.csv')
submission = pd.concat([submission.drop(columns=['Label']), pd.DataFrame(test_pred)], axis=1)
submission.columns = ['ID', 'Label']

submission.to_csv('submission.csv', index=False)
submission.head()
