import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from shopee.resnext import resnext269_face_mhe


class Linear_norm(nn.Module):
    def __init__(self, embed_size, num_classes, margin=0.7, scale=40):
        super(Linear_norm, self).__init__()
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(nn.init.kaiming_normal_(torch.Tensor(num_classes, embed_size)))

    def forward(self, x, y=None):
        logit = torch.mm(x, torch.t(F.normalize(self.weight)))
        out = logit[np.arange(logit.shape[0]), y]

        margin = torch.tensor(self.margin).cuda()
        cos_ = torch.cos(margin).cuda()
        sin_ = torch.sin(margin).cuda()

        if y is not None:
            logit[np.arange(logit.shape[0]), y] = (sin_ * out + cos_) * out - sin_

        logit = logit * self.scale
        return logit


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        cosine_all = F.linear(x, F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


class ShopeeModelTimm(nn.Module):
    def __init__(self, num_classes, backbone='tf_efficientnet_b1_ns'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=1)
        n_out = 1280
        n_embed = 512
        self.head = Linear_norm(n_embed, num_classes=num_classes, margin=0.5, scale=64)
        self.conv_last = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=7),
            nn.BatchNorm2d(n_out),
            nn.ReLU(),
        )
        self.embed = nn.Linear(n_out, n_embed, bias=False)

    def forward(self, x, y):
        x = self.backbone.forward_features(x)
        x = self.conv_last(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.embed(x)
        x = F.normalize(x)
        return self.head(x, y)


class ShopeeModelResnext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnext269_face_mhe(embedding_size=512)
        n_embed = 512
        self.head = Linear_norm(n_embed, num_classes=num_classes, margin=0.5, scale=64)

    def forward(self, x, y):
        x = self.backbone(x)
        return self.head(x, y)
