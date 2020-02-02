from collections import OrderedDict

import pretrainedmodels as pm
import torch
import torch.nn as nn
import torch.nn.functional as F

from bangali_19.beng_utils import get_head


def create_net(name, pretrained=True):
    if name == 'se_resnext50_32x4d':
        return pm.se_resnext50_32x4d()
    elif name == 'se_resnext101_32x4d':
        return pm.se_resnext101_32x4d()
    elif name == 'resnext101_32x8d_wsl':
        return torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    else:
        raise Exception('name ' + str(name) + ' is not supported')


class BengResnet(nn.Module):
    def forward(self, x):
        if self.input_bn:
            x = self.bn_in(x)

        x = self.layer0(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        if self.head == 'V0':
            x = F.adaptive_avg_pool2d(x, output_size=1)
            x = self.dropout_layer(x)
            x = x.view(x.size(0), -1)

            return self.cls1(x), self.cls2(x), self.cls3(x)
        elif self.isfoss_head:
            return self.cls1(x), self.cls2(x), self.cls3(x)
        else:
            return self.cls1(x), self.cls2(x), self.cls3(x)

    def __init__(self, name='se_resnext50_32x4d', pretrained=True, input_bn=True,
                 dropout=0.2, isfoss_head=False, head='V0'):
        super().__init__()
        self.head = head
        self.dropout = dropout
        self.input_bn = input_bn
        self.name = name
        self.isfoss_head = isfoss_head
        self.net = create_net(name=name, pretrained=pretrained)
        self.dropout_layer = nn.Dropout(p=dropout)

        layer0_modules = [('conv1',
                           nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                          ('bn1', nn.BatchNorm2d(64)),
                          ('relu1', nn.ReLU(inplace=True)),
                          ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        linear_size = {
            'se_resnext50_32x4d': 2048,
            'se_resnext101_32x4d': 2048,
            'resnext101_32x8d_wsl': 2048,
        }

        self.cls1, self.cls2, self.cls3 = get_head(isfoss_head, head, in_size=linear_size[name])

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
