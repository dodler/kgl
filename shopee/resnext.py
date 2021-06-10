import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shopee.mhe import ThomsonConv, ThomsonConvNoFinal, ThomsonLinear


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BottleneckMHE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32, dilation=1, final=False):
        super(BottleneckMHE, self).__init__()
        self.thomson_losses = []
        self.final = final

        if self.final:
            self.conv1 = ThomsonConv(inplanes, planes * 2, kernel_size=1, bias=False)
        else:
            self.conv1 = ThomsonConvNoFinal(inplanes, planes * 2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * 2)

        if self.final:
            self.conv2 = ThomsonConv(planes * 2, planes * 2, kernel_size=3, stride=stride,
                                     padding=1, bias=False, groups=num_group, dilation=dilation)
        else:
            self.conv2 = ThomsonConvNoFinal(planes * 2, planes * 2, kernel_size=3, stride=stride,
                                            padding=1, bias=False, groups=num_group, dilation=dilation)

        self.bn2 = nn.BatchNorm2d(planes * 2)

        if self.final:
            self.conv3 = ThomsonConv(planes * 2, planes * 4, kernel_size=1, bias=False)
        else:
            self.conv3 = ThomsonConvNoFinal(planes * 2, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ELU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        self.thomson_losses = []
        residual = x

        out, th = self.conv1(x)
        self.thomson_losses.append(th)
        out = self.bn1(out)
        out = self.relu(out)

        out, th = self.conv2(out)
        self.thomson_losses.append(th)
        out = self.bn2(out)
        out = self.relu(out)

        out, th = self.conv3(out)
        self.thomson_losses.append(th)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            # self.thomson_losses.append(th)

        out += residual
        out = self.relu(out)

        th = sum(self.thomson_losses)

        return out, th


class CustomForward(nn.Module):
    def __init__(self, modules):
        super(CustomForward, self).__init__()
        for i in range(len(modules)):
            self.add_module(str(i), modules[i])
        self.m_len = len(modules)
        self.th_losses = []

    def forward(self, x):
        self.th_losses = []
        for i in range(self.m_len):
            mod = self._modules[str(i)]
            if isinstance(mod, (BottleneckMHE, ThomsonConvNoFinal, ThomsonConv, ThomsonLinear)):
                x, th = mod(x)
                self.th_losses.append(th)
            else:
                x = mod(x)

        return x, sum(self.th_losses)


class ResNeXtMHE(nn.Module):

    def __init__(self, block, layers, num_group=32, wide=32, final=True, embedding_size=128):
        super(ResNeXtMHE, self).__init__()

        self.thomson_losses = []
        self.final = final
        self.inplanes = 64
        self.bn_input = nn.BatchNorm2d(3)

        if self.final:
            self.conv1 = ThomsonConv(3, 64, kernel_size=3, stride=2, padding=1,
                                     bias=False)
        else:
            self.conv1 = ThomsonConvNoFinal(3, 64, kernel_size=3, stride=2, padding=1,
                                            bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, wide, layers[0], num_group, final=self.final)
        self.layer2 = self._make_layer(block, wide * 2, layers[1], num_group, stride=2, final=self.final)
        self.layer3 = self._make_layer(block, wide * 4, layers[2], num_group, stride=2, final=self.final)
        self.layer4 = self._make_layer(block, wide * 8, layers[3], num_group, stride=2, final=self.final)

        if self.final:
            self.conv_last = ThomsonConv(wide * 8 * block.expansion, wide * 8 * block.expansion, kernel_size=7,
                                         groups=wide * 8 * block.expansion)
        else:
            self.conv_last = ThomsonConvNoFinal(wide * 8 * block.expansion, wide * 8 * block.expansion, kernel_size=7,
                                                groups=wide * 8 * block.expansion)

        self.bn_conv_last = nn.BatchNorm1d(wide * 8 * block.expansion)

        self.embed = nn.Linear(wide * 8 * block.expansion, embedding_size, bias=False)

        self.__init_weights__()

    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1, dilation=1, final=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, num_group=num_group, dilation=dilation, final=final))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return CustomForward(layers)

    def forward(self, x):
        self.thomson_losses = []
        x = self.bn_input(x)
        x, th = self.conv1(x)
        self.thomson_losses.append(th)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, th = self.layer1(x)
        self.thomson_losses.append(th)
        x, th = self.layer2(x)
        self.thomson_losses.append(th)
        x, th = self.layer3(x)
        self.thomson_losses.append(th)
        x, th = self.layer4(x)
        self.thomson_losses.append(th)

        x = F.adaptive_avg_pool2d(x, 7)

        x, th = self.conv_last(x)
        self.thomson_losses.append(th)
        x = x.view(x.size(0), -1)
        x = self.bn_conv_last(x)
        x = self.relu(x)
        raw_x = self.embed(x)

        n = F.normalize(raw_x)
        return n


def resnext50_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 4, 6, 3], **kwargs)
    return model


def resnext60_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 8, 6, 3], **kwargs)
    return model


def resnext111_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 8, 23, 3], **kwargs)
    return model


def resnext121_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 12, 48, 3], **kwargs)
    return model


def resnext101_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 4, 23, 3], **kwargs)
    return model


def resnext140_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 16, 23, 3], **kwargs)
    return model


def resnext269_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 30, 48, 8], **kwargs)
    return model


def resnext414_face_mhe(**kwargs):
    model = ResNeXtMHE(BottleneckMHE, [3, 30, 96, 8], **kwargs)
    return model


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    net = resnext50_face_mhe(num_classes=20, num_group=32, test=False, final=False)
    net.cuda()
    out, vecs, th = net(torch.zeros(2, 3, 224, 224).cuda())
    print(th.device)

    print(th / 150)
