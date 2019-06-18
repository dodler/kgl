import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
from tqdm import *

from google_landmark_retrieval.arguments import get_parser
from google_landmark_retrieval.data import LandmarkDataset
from google_landmark_retrieval.hard_triplet_loss import HardTripletLoss
from torchvision.models import resnet18, resnet50
from google_landmark_retrieval.vladnet import NetVLAD, EmbedNet
import numpy as np

from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
from apex.parallel import DistributedDataParallel as DDP
from imgaug import augmenters as iaa
import imgaug as ia

from tensorboardX import SummaryWriter

parser = get_parser()
args = parser.parse_args()

writer = SummaryWriter(log_dir='/tmp/tb/google_landmark_retrieval')

img_size = 320

train_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        rotate=(-45, 45), # rotate by -45 to +45 degrees
        shear=(-16, 16), # shear by -16 to +16 degrees
        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL),
    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
    iaa.CropToFixedSize(320,320),
    iaa.PadToFixedSize(320, 320)
])

valid_aug = iaa.Sequential([
    iaa.CropToFixedSize(320,320),
    iaa.PadToFixedSize(320, 320)
])

encoder = resnet50(pretrained=True)
base_model = nn.Sequential(
    encoder.conv1,
    encoder.bn1,
    encoder.relu,
    encoder.maxpool,
    encoder.layer1,
    encoder.layer2,
    encoder.layer3,
    encoder.layer4,
)

dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)

# Define model for embedding
net_vlad = NetVLAD(dim=dim, alpha=1.0)
model = EmbedNet(base_model, net_vlad).cuda()

ds = LandmarkDataset('/home/lyan/Documents/kaggle_data/google_landmark_retrieval/',
                     '/var/ssd_1t/google_landmark_retrieval_train/', 'train.csv', train_aug, test=False)

lr_step = 10
device = 0
gamma = 0.1

train_loader = DataLoader(ds, num_workers=args.num_workers, batch_size=args.batch_size)
criterion = HardTripletLoss(margin=0.3).cuda()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_step, gamma=gamma)

if args.mixed:
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale)

# model = DDP(model, delay_allreduce=True)

cnt = 0

if args.resume is not None:
    pretrained_dict = torch.load(args.resume)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    try:
        model.load_state_dict(model_dict)
    except Exception as e:
        print('loading state dict, error:',e)

torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)

model = nn.DataParallel(model)


torch.save(model.state_dict(), 'checkpoints/resnet18_-1.pth')
for i in range(args.start_epoch, args.epochs):
    scheduler.step()

    t = tqdm(train_loader)

    epoch_start = time.time()

    model.train()

    for data in t:
        data_input, label, idx = data
        data_input = data_input.to(device)
        label = label.to(device)
        output = model(data_input)
        loss = criterion(output, label)

        t.set_description('loss:' + str(np.around(loss.item(), 4)))

        writer.add_scalar(args.name + '_loss/batch', loss.item(), cnt)

        optimizer.zero_grad()
        if args.mixed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        cnt += 1
    torch.save(model.module.state_dict(), 'checkpoints/' + args.name + '_epoch_' + str(i) + '.pth')
