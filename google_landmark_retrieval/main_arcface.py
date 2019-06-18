import time

import numpy as np
import torch
import torch.nn as nn
from apex import amp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import *

from google_landmark_retrieval.arcface import FocalLoss, ArcMarginProduct, AddMarginProduct
from google_landmark_retrieval.arguments import get_parser
from google_landmark_retrieval.aug import get_augs
from google_landmark_retrieval.constants import IMG_SIZE, FEATURE_SIZE
from google_landmark_retrieval.data import LandmarkDataset
from google_landmark_retrieval.resnet_face import resnet_arcface
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os.path as osp
import os

CHECKPOINTS_PATH = '/home/lyan/checkpoints'

parser = get_parser()
args = parser.parse_args()

lr_step = 20
device = 0
gamma = 0.1

writer = SummaryWriter(log_dir='/tmp/tb/google_landmark_retrieval')

model = resnet_arcface().to(device)
metric_fc = ArcMarginProduct(FEATURE_SIZE, args.num_classes, m=0.3).to(device)

train_aug, valid_aug = get_augs(img_size=IMG_SIZE)

train_ds = LandmarkDataset('/home/lyan/Documents/kaggle_data/google_landmark_retrieval/',
                           '/var/ssd_1t/google_landmark_retrieval_train/', 'train1.csv', train_aug)

valid_ds = LandmarkDataset('/home/lyan/Documents/kaggle_data/google_landmark_retrieval/',
                           '/var/ssd_1t/google_landmark_retrieval_train/', 'val.csv', valid_aug)

train_loader = DataLoader(train_ds, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)
valid_loader = DataLoader(valid_ds, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                          pin_memory=True)

if args.crit == 'focal':
    criterion = FocalLoss(gamma=2).to(device)
else:
    criterion = torch.nn.CrossEntropyLoss()

cnt = 0
epoch_start = 0


def restore_model(checkpoint_path):
    global epoch_start
    state = torch.load(checkpoint_path)
    print(f'restoring from state with loss {state["val_loss"]} and acc {state["val_acc"]}')
    model.load_state_dict(state['model_0'])
    metric_fc.load_state_dict(state['model_1'])

    # optimizer.load_state_dict(state['opt'])
    epoch_start = int(state['epoch'])


# optimizer = torch.optim.Adam(list(model.parameters()) + list(metric_fc.parameters()), lr=args.lr)
optimizer = torch.optim.SGD(list(model.parameters()) + list(metric_fc.parameters()), lr=args.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step, gamma=gamma)

if args.resume is not None:
    restore_model(args.resume)

# if args.mixed:
#     model, optimizer = amp.initialize(model, optimizer,
#                                       opt_level=args.opt_level,
#                                       keep_batchnorm_fp32=args.keep_batchnorm_fp32,
#                                       loss_scale=args.loss_scale)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    metric_fc = nn.DataParallel(metric_fc)

# if args.clip is not None:
#     torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip)


def get_acc(raw_output, gt):
    with torch.no_grad():
        pred = F.softmax(raw_output, dim=1)
        pred = torch.argmax(pred, dim=1).detach().cpu().numpy()

        return accuracy_score(gt.detach().cpu().numpy(), pred)


def save_state(models, optimizer, name, epoch=-1, val_loss=0, val_acc=0):
    state = {
        'epoch': epoch,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'opt': optimizer.state_dict()
    }

    for i in range(len(models)):
        if isinstance(models[i], nn.DataParallel):
            model_state = models[i].module.state_dict()
        else:
            model_state = models[i].state_dict()
        state['model_' + str(i)] = model_state

    if not osp.exists(CHECKPOINTS_PATH):
        os.mkdir(CHECKPOINTS_PATH)

    checkpoint_name = name + '_' + str(epoch) + '.pth'
    checkpoint_name = osp.join(CHECKPOINTS_PATH, checkpoint_name)
    torch.save(state, checkpoint_name)


def validate(data_loader):
    model.eval()
    metric_fc.eval()

    scheduler.step()

    t = tqdm(data_loader)

    losses = []
    scores = []

    for data in t:
        data_input, label, _ = data
        data_input = data_input.to(device)
        label = label.to(device)

        with torch.no_grad():
            feature = model(data_input)
            outputs = metric_fc(feature, label)

            loss = criterion(outputs, label)

            acc = get_acc(outputs, label)

        losses.append(loss.item())
        scores.append(acc)

    return np.array(losses).mean(), np.array(scores).mean()


print('optimizer lr')
for param_group in optimizer.param_groups:
    print(param_group['lr'])
print('using crit', args.crit)

for i in range(epoch_start, epoch_start + args.epochs):
    scheduler.step()

    t = tqdm(train_loader)

    epoch_start = time.time()

    model.train()
    metric_fc.train()

    for data in t:
        data_input, label, _ = data
        data_input = data_input.to(device)
        label = label.to(device)

        feature = model(data_input)
        outputs = metric_fc(feature, label)
        acc = get_acc(outputs, label)

        loss = criterion(outputs, label)

        optimizer.zero_grad()
        if args.mixed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        t.set_description('loss:' + str(np.around(loss.item(), 4)) + ' acc:' + str(np.around(acc, 4)))

        writer.add_scalar(args.name + '_loss/batch', loss.item(), cnt)
        writer.add_scalar(args.name + '_acc/batch', acc, cnt)
        if cnt % 100 == 0:
            writer.add_images(args.name + '_image/batch', data_input[0:16, :, :, :])

        cnt += 1

    val_loss, val_acc = validate(valid_loader)
    save_state([model, metric_fc], optimizer, args.name, i)
    print('validation', val_loss, val_acc)
