from fastai.vision import *
import torch.nn as nn
import torch

from kaggle_lyan_utils import Mish

arch = models.densenet121
nunique = [168, 11, 7]


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        layers = [AdaptiveConcatPool2d(), Mish(), Flatten()] + \
                 bn_drop_lin(nc * 2, 512, True, ps, Mish()) + \
                 bn_drop_lin(512, n, True, ps)
        self.fc = nn.Sequential(*layers)
        #         self.fc = nn.Sequential(AdaptiveConcatPool2d(), Flatten(), nn.Linear(nc*2,n), )
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


# change the first conv to accept 1 chanel input
class Dnet_1ch(nn.Module):
    def __init__(self, arch=arch, n=nunique, pre=True, ps=0.5):
        super().__init__()
        m = arch(True) if pre else arch()

        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        w = (m.features.conv0.weight.sum(1)).unsqueeze(1)
        conv.weight = nn.Parameter(w)

        self.layer0 = nn.Sequential(conv, m.features.norm0, nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            m.features.denseblock1)
        self.layer2 = nn.Sequential(m.features.transition1, m.features.denseblock2)
        self.layer3 = nn.Sequential(m.features.transition2, m.features.denseblock3)
        self.layer4 = nn.Sequential(m.features.transition3, m.features.denseblock4,
                                    m.features.norm5)

        nc = self.layer4[-1].weight.shape[0]
        self.head1 = Head(nc, n[0])
        self.head2 = Head(nc, n[1])
        self.head3 = Head(nc, n[2])
        # to_Mish(self.layer0), to_Mish(self.layer1), to_Mish(self.layer2)
        # to_Mish(self.layer3), to_Mish(self.layer4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1, x2, x3
