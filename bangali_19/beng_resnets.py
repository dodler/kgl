from collections import OrderedDict

import pretrainedmodels as pm
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai import AdaptiveConcatPool2d, Flatten, bn_drop_lin

from kaggle_lyan_utils import Mish


def create_net(name, pretrained=True):
    if name == 'se_resnext50_32x4d':
        if pretrained:
            return pm.se_resnext50_32x4d()
        else:
            return pm.se_resnext50_32x4d(pretrained=None)
    elif name == 'se_resnext101_32x4d':
        if pretrained:
            return pm.se_resnext101_32x4d()
        else:
            return pm.se_resnext101_32x4d()
    raise Exception('name ' + str(name) + ' is not supported')


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


class BengResnet(nn.Module):
    def forward(self, x):
        if self.input_bn:
            x = self.bn_in(x)

        x = self.layer0(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        if self.net.dropout is not None:
            x = self.net.dropout(x)
        x = x.view(x.size(0), -1)

        return self.cls1(x), self.cls2(x), self.cls3(x)

    def __init__(self, name='se_resnext50_32x4d', pretrained=True, input_bn=True,
                 dropout=0.2, isfoss_head=False):
        super().__init__()
        self.dropout = dropout
        self.input_bn = input_bn
        self.name = name
        self.net = create_net(name=name, pretrained=pretrained)
        # self.net.dropout = nn.Dropout(p=dropout)

        # fixme, replace 64 with value from net
        layer0_modules = [('conv1',
                           nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                          ('bn1', nn.BatchNorm2d(64)),
                          ('relu1', nn.ReLU(inplace=True)),
                          ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        linear_size = {
            'se_resnext50_32x4d': 2048,
            'se_resnext101_32x4d': 2048,
        }

        if isfoss_head:
            self.cls1 = Head(linear_size[name], 168)
            self.cls2 = Head(linear_size[name], 11)
            self.cls3 = Head(linear_size[name], 7)
        else:
            self.cls1 = nn.Linear(linear_size[name], 168)
            self.cls2 = nn.Linear(linear_size[name], 11)
            self.cls3 = nn.Linear(linear_size[name], 7)

        if input_bn:
            self.bn_in = nn.BatchNorm2d(1)


if __name__ == '__main__':
    net = BengResnet(name='se_resnext50_32x4d')
    print(net.net)

    inp = torch.randn(2, 1, 224, 224)
    x1, x2, x3 = net(inp)
    print(x1.shape, x2.shape, x3.shape)

    inp = torch.randn(2, 1, 128, 128)
    x1, x2, x3 = net(inp)
    print(x1.shape, x2.shape, x3.shape)
