from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai import AdaptiveConcatPool2d, Flatten, bn_drop_lin

from bangali_19.beng_heads import Head
from bangali_19.beng_utils import get_head
from cust_densenet import densenet201, densenet169, densenet161, densenet121
from kaggle_lyan_utils import Mish


def create_net(name):
    nets = {
        'densenet121': densenet121,
        'densenet161': densenet161,
        'densenet169': densenet169,
        'densenet201': densenet201
    }

    if name in nets:
        return nets[name]()
    else:
        raise Exception('name ' + str(name) + ' is not supported')


class BengDensenet(nn.Module):
    def forward(self, x):
        if self.input_bn:
            x = self.bn_in(x)

        x = self.net.conv_in(x)
        x = self.net.layers(x)

        if self.isfoss_head:
            return self.cls1(x), self.cls2(x), self.cls3(x)
        else:
            x = F.adaptive_avg_pool2d(x, output_size=1)
            if self.dropout is not None:
                x = self.dropout_layer(x)
            x = x.view(x.size(0), -1)

            return self.cls1(x), self.cls2(x), self.cls3(x)

    def __init__(self, name='densenet121', input_bn=True,
                 dropout=0.2, isfoss_head=False, head='V0'):
        super().__init__()
        self.head = head
        if dropout is not None:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None
        self.dropout = dropout
        self.input_bn = input_bn
        self.name = name
        self.isfoss_head = isfoss_head
        self.net = create_net(name=name)

        linear_size = {
            'densenet121': 1024,
            'densenet161': 2208,
            'densenet169': 1664,
            'densenet201': 1920,
        }

        self.cls1, self.cls2, self.cls3 = get_head(isfoss_head, head, in_size=linear_size[name])

        if input_bn:
            self.bn_in = nn.BatchNorm2d(1)


if __name__ == '__main__':
    net = BengDensenet(name='densenet201')
    print(net.net)

    inp = torch.randn(2, 1, 224, 224)
    x1, x2, x3 = net(inp)
    print(x1.shape, x2.shape, x3.shape)

    inp = torch.randn(2, 1, 128, 128)
    x1, x2, x3 = net(inp)
    print(x1.shape, x2.shape, x3.shape)
