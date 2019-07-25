import argparse
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import *

from siim_acr_pnuemotorax.siim_data import SIIMDatasetSegmentation

parser = argparse.ArgumentParser(description='SIIM ACR Pneumotorax unet training')

parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--opt-step-size', default=10, type=int)
parser.add_argument('--resume', type=str, default=None)

args = parser.parse_args()

if args.resume is not None:
    model = torch.load(args.resume)

model.to(0)
model.eval()

sub=pd.read_csv('/home/lyan/Documents/kaggle/siim_acr_pnuemotorax/subm1118.csv')
img_ids=sub.ImageId.values.tolist()
ds = SIIMDatasetSegmentation(image_dir='/var/ssd_1t/siim_acr_pneumo/test_png', img_ids=img_ids, mask_dir=None, aug=None)
loader = DataLoader(ds, shuffle=False, batch_size=args.batch_size,
                    num_workers=8, drop_last=False)
predictions = []
for im in tqdm(loader):
    im = im.to(0)
    with torch.no_grad():
        pred = model(im).detach().cpu().numpy()
        predictions.append(pred)

with open('result.pkl', 'wb') as f:
    pickle.dump(predictions, f)

print('done')
