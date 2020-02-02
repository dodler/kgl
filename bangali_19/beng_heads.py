import torch
from fastai import AdaptiveConcatPool2d, Flatten, bn_drop_lin
import torch.nn as nn
import torch.nn.functional as F
from kaggle_lyan_utils import Mish


class Head(nn.Module):
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
        return self.fc(x)