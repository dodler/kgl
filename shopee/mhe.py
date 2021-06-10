import math

import numpy as np
import torch.nn as nn
import torch


def print_stat(x):
    print(x.shape, x.mean(), x.std())


class Thomson(nn.Module):
    def __init__(self, n_filt, power, model='half_mhe'):
        '''
        :param n_filt:
        :param power:
        :param model: mhe or half_mhe
        '''
        super(Thomson, self).__init__()
        self.model = model
        self.n_filt = n_filt
        self.power = power

    def forward(self, filt):
        n_filt = self.n_filt
        model = self.model
        power = self.power

        filt = filt.view(-1, n_filt)

        if model == 'half_mhe':
            filt_neg = filt * -1
            filt = torch.cat((filt, filt_neg), dim=1)
            n_filt *= 2

        filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True)) + 1e-4

        norm_mat = torch.mm(filt_norm.t(), filt_norm) + 1e-6

        inner_pro = torch.mm(filt.t(), filt)

        inner_pro /= norm_mat

        if power == '0':
            ones = torch.ones(n_filt).to(filt.device)
            cross_terms = 2.0 - 2.0 * inner_pro
            final = -torch.log(cross_terms + torch.diag(ones))
            final -= torch.triu(final, -1)
            loss = 1 * final.sum() / (1e-8 + n_filt * (n_filt - 1) / 2.0)
        elif power == '1':
            ones = torch.ones(n_filt).to(filt.device)
            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(ones))
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 1 * final.sum() / cnt
        elif power == '2':
            ones = torch.ones(n_filt).to(filt.device)
            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(ones))
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-1))
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 1 * final.sum() / cnt
        elif power == 'a0':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = -torch.log(acos)
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 1 * final.sum() / cnt
        elif power == 'a1':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = torch.pow(acos, torch.ones_like(acos) * (-1))
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 1e-1 * final.sum() / cnt
        elif power == 'a2':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = torch.pow(acos, torch.ones_like(acos) * (-2))
            t = torch.triu(final, -1)
            final = final - t  # torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 1e-1 * final.sum() / cnt
        return loss


class ThomsonFinal(nn.Module):

    def __init__(self, n_filt, power):
        super(ThomsonFinal, self).__init__()
        self.power = power
        self.n_filt = n_filt

    def forward(self, filt):
        n_filt = self.n_filt
        power = self.power

        filt = filt.view(-1, n_filt)

        filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True)) + 1e-4
        norm_mat = torch.mm(filt_norm.t(), filt_norm) + 1e-6
        inner_pro = torch.mm(filt.t(), filt)
        inner_pro /= norm_mat

        if power == '0':
            cross_terms = 2.0 - 2.0 * inner_pro
            ones = torch.ones(n_filt).to(filt.device)
            final = -torch.log(cross_terms + torch.diag(ones))
            final -= torch.triu(final, -1)
            loss = 10 * final.sum() / (1e-8 + n_filt * (n_filt - 1) / 2.0)
        elif power == '1':
            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.ones(n_filt)))
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 10 * final.sum() / cnt
        elif power == '2':
            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.ones(n_filt)))
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-1))
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 10 * final.sum() / cnt
        elif power == 'a0':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = -torch.log(acos)
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 10 * final.sum() / cnt
        elif power == 'a1':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = torch.pow(acos, torch.ones_like(acos) * (-1))
            final -= torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = 1 * final.sum() / cnt
        elif power == 'a2':
            acos = (torch.acos(inner_pro) / math.pi) + 1e-4
            final = torch.pow(acos, torch.ones_like(acos) * (-2))
            t = torch.triu(final, -1)
            final = final - t  # torch.triu(final, -1)
            cnt = n_filt * (n_filt - 1) / 2.0 + 1e-8
            loss = final.sum() / cnt
        return loss


class ThomsonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, power='a2', final=True):
        super(ThomsonLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if final:
            self.thomson = ThomsonFinal(out_features, power)
        else:
            self.thomson = Thomson(out_features, power)

    def forward(self, x):
        if self.training:
            th = self.thomson(self.linear.weight)
        else:
            th = 0
        return self.linear(x), th


class ThomsonConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, power='a2'):
        '''
        convolution with final thomson ( a bit larger constatns)
        :param in_channels:
        :param out_channels: aka n_filt from the original implementation
        :param power:
        :param activation: activation function
        '''
        super(ThomsonConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.thomson = ThomsonFinal(out_channels, power)
        torch.nn.init.normal_(self.conv.weight)

    def forward(self, x):
        # print('conv filter shape',self.conv.weight.shape)
        if self.training:
            thom = self.thomson(self.conv.weight)
        else:
            thom = 0
        x = self.conv(x)
        return x, thom


class ThomsonConvNoFinal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, power='a2', model='half_mhe'):
        '''
        convolution with thomson
        :param in_channels:
        :param out_channels: aka n_filt from the original implementation
        :param power:
        :param activation: activation function
        '''
        super(ThomsonConvNoFinal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.conv_filt_len = 1
        sh = self.conv.weight.shape
        for s in sh:
            self.conv_filt_len *= s

        self.thomson = Thomson(out_channels, power, model=model)
        torch.nn.init.normal_(self.conv.weight)

    def forward(self, x):
        if self.training:
            thom = self.thomson(self.conv.weight)
        else:
            thom = 0
        x = self.conv(x)
        return x, thom


class TestFullyConnected(nn.Module):
    def __init__(self):
        super(TestFullyConnected, self).__init__()
        self.conv1 = ThomsonConv(3, 32, kernel_size=3)
        self.conv2 = ThomsonConv(32, 64, kernel_size=3)
        self.conv3 = ThomsonConv(64, 1, kernel_size=12)

    def forward(self, x):
        x, t1 = self.conv1(x)
        x, t2 = self.conv2(x)
        x, t3 = self.conv3(x)

        # print(t1, t2, t3)
        return x, (t1 + t2 + t3) / 3


class VGGTorch():

    def _add_thomson_constraint_final(self, filt, n_filt, power):
        filt = torch.reshape(filt, [-1, n_filt])
        filt_norm = torch.sqrt(torch.mean(filt * filt, [0], keep_dims=True) + 1e-4)
        norm_mat = torch.mm(torch.transpose(filt_norm), filt_norm)
        inner_pro = torch.mm(torch.transpose(filt), filt)
        inner_pro /= norm_mat

        if power == '0':
            cross_terms = 2.0 - 2.0 * inner_pro
            final = -torch.log(cross_terms + torch.diag([1.0] * n_filt))
            final -= torch.triu(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * torch.mean(final) / cnt
        elif power == '1':
            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag([1.0] * n_filt))
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-0.5))
            final -= torch.triu(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * torch.mean(final) / cnt
        elif power == '2':
            cross_terms = (2.0 - 2.0 * inner_pro + torch.diag([1.0] * n_filt))
            final = torch.pow(cross_terms, torch.ones_like(cross_terms) * (-1))
            final -= torch.triu(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * torch.mean(final) / cnt
        elif power == 'a0':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = -torch.log(acos)
            final -= torch.triu(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * torch.mean(final) / cnt
        elif power == 'a1':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = torch.pow(acos, torch.ones_like(acos) * (-1))
            final -= torch.triu(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1 * torch.mean(final) / cnt
        elif power == 'a2':
            acos = torch.acos(inner_pro) / math.pi
            acos += 1e-4
            final = torch.pow(acos, torch.ones_like(acos) * (-2))
            final -= torch.triu(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1 * torch.mean(final) / cnt
        return loss

    def _conv_layer(self, bottom, ksize, n_filt, name, stride=1,
                    pad='SAME', relu=False, reg=True, bn=True, model='baseline', power='0', final=False):

        n_input = bottom.get_shape().as_list()[3]
        shape = [ksize, ksize, n_input, n_filt]
        # print("shape of filter %s: %s" % (name, str(shape)))

        filt = self.get_conv_filter(shape, reg, stddev=torch.sqrt(2.0 / (ksize * ksize * n_input).float()))
        if model == 'mhe' or model == 'half_mhe':
            if final:
                self._add_thomson_constraint_final(filt, n_filt, power)
            else:
                self._add_thomson_constraint(filt, n_filt, model, power)

        conv = torch.nn.Conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)
        ## fixme

        if bn:
            conv = torch.nn.BatchNorm2d(n_filt)

        if relu:
            return torch.nn.ReLU(conv)
        else:
            return conv

    def build(self, rgb, n_class, model_name, power_s):
        self.wd = 5e-4

        feat = (rgb - 127.5) / 128.0

        ksize = 3
        n_layer = 3

        # 32X32
        n_out = 64
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, name="conv1_" + str(i), bn=True, relu=True,
                                    pad='SAME', reg=True, model=model_name, power=power_s)
        feat = self._max_pool(feat, 'pool1')

        # 16X16
        n_out = 128
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, name="conv2_" + str(i), bn=True, relu=True,
                                    pad='SAME', reg=True, model=model_name, power=power_s)
        feat = self._max_pool(feat, 'pool2')

        # 8X8
        n_out = 256
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, name="conv3_" + str(i), bn=True, relu=True,
                                    pad='SAME', reg=True, model=model_name, power=power_s)
        feat = self._max_pool(feat, 'pool3')

        self.fc6 = self._conv_layer(feat, 4, 256, "fc6", bn=False, relu=False, pad='VALID',
                                    reg=True, model=model_name, power=power_s)

        self.score = self._conv_layer(self.fc6, 1, n_class, "score", bn=False, relu=False, pad='VALID',
                                      reg=True, model=model_name, power=power_s, final=True)

        self.pred = torch.squeeze(torch.argmax(self.score, axis=3))

    def build_test_conv_mhe_32(self, rgb):
        self.wd = 5e-4
        feat = (rgb - 127.5) / 128.0

        return self._conv_layer(feat, 3, 32, name="conv1", bn=True, relu=True,
                                pad='SAME', reg=True, model='mhe', power='0')


if __name__ == '__main__':
    n_class = 100
    batch_sz = 2
    batch_test = 100
    max_epoch = 42500
    lr = 1e-3
    momentum = 0.9

    t = ThomsonConvNoFinal(3, 64, 3, power='0', model='mhe')
    inp = np.random.uniform(0, 1, (90, 3, 32, 32)).astype(np.float32)
    with torch.no_grad():
        conv_prod, thom_loss = t(torch.from_numpy(inp))
        print_stat(conv_prod)
        print(thom_loss)
