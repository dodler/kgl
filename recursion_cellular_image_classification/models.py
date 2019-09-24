import sys

from efficientnet_pytorch import EfficientNet

sys.path.append('/home/lyan/Documents/enorm/enorm')
sys.path.append('/home/lyan/Documents/rxrx1-utils')

PRINT_FREQ = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
# from enorm import ENorm
import warnings

warnings.filterwarnings('ignore')

import pretrainedmodels as pm


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class ArcSEResnext50(nn.Module):
    def __init__(self, num_classes):
        super(ArcSEResnext50, self).__init__()
        self.feature_extr = pm.se_resnext50_32x4d()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.metric = ArcMarginProduct(in_features=2048, out_features=num_classes)

        weights = torch.zeros(64, 6, 7, 7, dtype=torch.float32)
        weights[:, 0, :, :] = self.feature_extr.layer0.conv1.weight[:, 0, :, :]
        weights[:, 1, :, :] = self.feature_extr.layer0.conv1.weight[:, 0, :, :]
        weights[:, 2, :, :] = self.feature_extr.layer0.conv1.weight[:, 1, :, :]
        weights[:, 3, :, :] = self.feature_extr.layer0.conv1.weight[:, 1, :, :]
        weights[:, 4, :, :] = self.feature_extr.layer0.conv1.weight[:, 2, :, :]
        weights[:, 5, :, :] = self.feature_extr.layer0.conv1.weight[:, 2, :, :]

        self.feature_extr.layer0.conv1 = nn.Conv2d(6, 64, (7, 7), (2, 2), (3, 3), bias=False)
        self.feature_extr.layer0.conv1.weight = torch.nn.Parameter(weights)

    def upd_metric(self, metric):
        self.metric = metric

    def forward(self, x, labels):
        bs = x.shape[0]
        x = self.feature_extr.features(x)
        x = self.pool(x).reshape(bs, -1)

        if labels is not None:
            return self.metric(x, labels)
        else:
            return F.normalize(x)


class ArcEffNetb0(nn.Module):
    def __init__(self, num_classes=1108):
        super(ArcEffNetb0, self).__init__()
        self.num_classes = num_classes
        self.feature_extr = EfficientNet.from_pretrained('efficientnet-b0')

        weights = torch.zeros(32, 6, 3, 3, dtype=torch.float32)
        weights[:, 0, :, :] = self.feature_extr._conv_stem.weight[:, 0, :, :]
        weights[:, 1, :, :] = self.feature_extr._conv_stem.weight[:, 0, :, :]
        weights[:, 2, :, :] = self.feature_extr._conv_stem.weight[:, 1, :, :]
        weights[:, 3, :, :] = self.feature_extr._conv_stem.weight[:, 1, :, :]
        weights[:, 4, :, :] = self.feature_extr._conv_stem.weight[:, 2, :, :]
        weights[:, 5, :, :] = self.feature_extr._conv_stem.weight[:, 2, :, :]

        self.feature_extr._conv_stem = nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.feature_extr._conv_stem.weight = torch.nn.Parameter(weights)

    def forward(self, x):
        bs = x.shape[0]
        x = self.feature_extr.extract_features(x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).reshape(bs, -1)
        return F.normalize(x)


class ArcResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ArcResNet34, self).__init__()
        self.num_classes = num_classes
        self.feature_extr = torchvision.models.resnet34()
        self.metric = ArcMarginProduct(in_features=512, out_features=num_classes)

        weights = torch.zeros(64, 6, 7, 7, dtype=torch.float32)
        weights[:, 0, :, :] = self.feature_extr.conv1.weight[:, 0, :, :]
        weights[:, 1, :, :] = self.feature_extr.conv1.weight[:, 0, :, :]
        weights[:, 2, :, :] = self.feature_extr.conv1.weight[:, 1, :, :]
        weights[:, 3, :, :] = self.feature_extr.conv1.weight[:, 1, :, :]
        weights[:, 4, :, :] = self.feature_extr.conv1.weight[:, 2, :, :]
        weights[:, 5, :, :] = self.feature_extr.conv1.weight[:, 2, :, :]

        self.feature_extr.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extr.conv1.weight = torch.nn.Parameter(weights)

    def feats(self,x):
        x = self.feature_extr.conv1(x)
        x = self.feature_extr.bn1(x)
        x = self.feature_extr.relu(x)
        x = self.feature_extr.maxpool(x)

        x = self.feature_extr.layer1(x)
        x = self.feature_extr.layer2(x)
        x = self.feature_extr.layer3(x)
        x = self.feature_extr.layer4(x)

        x = self.feature_extr.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x, labels):
        x = self.feats(x)
        return x