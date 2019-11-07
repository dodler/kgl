import argparse
import datetime
import sys
from collections import OrderedDict
from catalyst.dl.runner import SupervisedRunner
import numpy as np
import pandas as pd
from catalyst.dl.callbacks import EarlyStoppingCallback, AccuracyCallback

from severstal.sev_data import SevPretrain

SEED = 42
from catalyst.utils import set_global_seed, prepare_cudnn

set_global_seed(SEED)
prepare_cudnn(deterministic=True)

import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

sys.path.append('/home/lyan/Documents/Synchronized-BatchNorm-PyTorch')

NON_BEST_DONE_THRESH = 15

parser = argparse.ArgumentParser(description='Severstal segmentation train')

parser.add_argument('--backbone',required=False, default='efficientnet-b0')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/severstal/train/', required=False)
args = parser.parse_args()

model = EfficientNet.from_pretrained(args.backbone)
model._fc = torch.nn.Linear(1280, 20)
torch.nn.init.xavier_normal_(model._fc.weight)

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

lr = 0.0001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
loaders = OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = train_loader

num_epochs = args.epochs
logdir = "/var/data/logs/" + experiment_name
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    callbacks=[
        AccuracyCallback(accuracy_args=[1]),
        EarlyStoppingCallback(patience=10, min_delta=0.01)
    ],
    verbose=True
)