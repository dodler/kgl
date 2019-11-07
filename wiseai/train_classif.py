from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import torch
import torchvision
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback, AccuracyCallback, AUCCallback
import argparse
import torch.optim as optim

from wiseai.augs import train_aug, valid_aug
from wiseai.models import SkinSEResnext50

dataset_path = '/content/gdrive/My Drive/Train v22.05/'

parser = argparse.ArgumentParser(description='Nevus classification model training')

parser.add_argument('--lr',
                    default=2e-5,
                    type=float,
                    help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=14)
parser.add_argument('--epochs', type=int, default=4)
# parser.add_argument('--image-dir', type=str, default='/var/ssd_1t/severstal/train/', required=False)
parser.add_argument('--folds-path', type=str, default='stage_1_train_folds.csv',
                    required=False)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()


train_ds = torchvision.datasets.ImageFolder('skin_data/train', transform=train_aug)
valid_ds = torchvision.datasets.ImageFolder('skin_data/val', transform=valid_aug)

train_loader = torch.utils.data.DataLoader(train_ds, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)
valid_loader = torch.utils.data.DataLoader(valid_ds, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size)

num_classes = len(train_ds.classes)
print(num_classes)
model = SkinSEResnext50(num_classes)
model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
crit=torch.nn.CrossEntropyLoss()

plist = [{'params': model.parameters(), 'lr': args.lr}]
optimizer = optim.Adam(plist, lr=args.lr)

experiment_name = 'cls_' + args.model + '_' + '_f_'+str(args.fold)+'_' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
experiment_name = experiment_name.replace('/', '_')

lr = args.lr

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, mode='min')
criterion = torch.nn.BCEWithLogitsLoss().float()
loaders = OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader

num_epochs = args.epochs
logdir = "/var/data/wiseai_cls_logs/" + experiment_name
runner = SupervisedRunner()


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
        AccuracyCallback(num_classes=num_classes),
        AUCCallback(
            num_classes=num_classes,
            input_key="targets_one_hot",
            class_names=class_names
        ),
        EarlyStoppingCallback(patience=5, min_delta=0.01)
    ],
    verbose=True
)
