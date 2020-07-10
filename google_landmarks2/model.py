import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import numpy as np
from torch.nn import Parameter
import math
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device='cuda'):
        super(ArcMarginProduct, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class EffB0Arc(nn.Module):

    def __init__(self, n_class, test=False, device='cpu'):
        super(EffB0Arc, self).__init__()

        self.test = test
        self.back = EfficientNet.from_pretrained('efficientnet-b0')
        self.embed = nn.Linear(1280, 128, bias=False)
        print('effnet usng device', device)
        self.arc = ArcMarginProduct(128, n_class, device=device)
        self.bn_vec = nn.BatchNorm1d(128)
        self.bn_feat_out = nn.BatchNorm1d(1280)
        self.dropout_feat = nn.Dropout()

    def forward(self, x, label):
        x = self.back.extract_features(x)
        x = self.back._avg_pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.bn_feat_out(x)
        x = self.dropout_feat(x)
        vec = self.embed(x)
        if self.test:
            return F.normalize(vec)
        else:
            return self.arc(vec, label), vec


if __name__ == '__main__':
    net = EffB0Arc(n_class=100)
    net.eval()
    b = 24
    cls, vec = net(torch.randn(b, 3, 224, 224), torch.ones(b, 1))
    print(cls.shape, vec.shape)
