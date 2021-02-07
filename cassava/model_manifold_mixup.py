import torch.nn as nn
import timm
import torch.nn.functional as F
import torch
import numpy as np


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)

    # t1 = target.data.cpu().numpy()
    # t2 = target[indices].data.cpu().numpy()
    # print (np.sum(t1==t2))
    return out, target_reweighted


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)

    return y_onehot.cuda()


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


class CassavaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = timm.create_model(cfg['backbone'], pretrained=True)
        if 'efficient' in cfg['backbone']:
            n_features = backbone.classifier.in_features
        else:
            n_features = backbone.fc.in_features
        self.backbone = backbone
        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x, y=None):
        # works only for efn

        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)

        if y is not None:
            target_reweighted = to_one_hot(y, 5)
        else:
            target_reweighted = None

        if y is not None:
            lam = get_lambda(alpha=2.0)  # from github
            x, target_reweighted = mixup_process(out=x, target_reweighted=target_reweighted, lam=lam)

        x = self.backbone.blocks(x)

        if y is not None:
            lam = get_lambda(alpha=2.0)  # from github
            x, target_reweighted = mixup_process(out=x, target_reweighted=target_reweighted, lam=lam)

        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        feats = self.backbone.act2(x)

        x = self.pool(feats).view(x.size(0), -1)

        x = self.classifier(x)

        return x, feats, target_reweighted
