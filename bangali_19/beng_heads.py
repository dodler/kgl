import torch
from fastai.vision import *
import torch.nn as nn
import torch.nn.functional as F
from kaggle_lyan_utils import Mish


class HeadV3(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        layers = [AdaptiveConcatPool2d(), Mish(), Flatten()] + \
                 bn_drop_lin(nc * 2, 512, True, ps, Mish()) + \
                 bn_drop_lin(512, n, True, ps)
        self.fc = nn.Sequential(*layers)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


class HeadV1(nn.Module):
    def __init__(self, inp, outp, dropout=0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=inp)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.lin = nn.Linear(in_features=inp, out_features=outp)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.lin(x)
        return x


class HeadV2(nn.Module):
    def __init__(self, inp, outp, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(num_features=inp)
        self.drop1 = nn.Dropout(p=dropout, inplace=True)
        self.lin1 = nn.Linear(in_features=inp, out_features=512)

        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.drop2 = nn.Dropout(p=dropout, inplace=True)
        self.lin2 = nn.Linear(in_features=512, out_features=outp)

        self.act = Mish()

        self.head_layers = nn.Sequential(
            self.bn1, self.drop1, self.lin1, self.act, self.bn2, self.drop2, self.lin2
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        # x = F.adaptive_avg_pool2d(x, output_size=1)
        # x = x.view(x.size(0), -1)
        return self.head_layers(x)
