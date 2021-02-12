import random

import timm
import torch.nn as nn

from cassava.pazzle_mix.pazzle_mixup import mixup_process, to_one_hot, get_lambda


class CassavaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = timm.create_model('seresnext50_32x4d', pretrained=True)
        n_features = backbone.fc.in_features
        self.backbone = backbone
        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x, y=None, grad=None):
        layer_mix = random.randint(0, 2)
        if y is not None:
            target_reweighted = to_one_hot(y, 5)
        else:
            target_reweighted = None

        if layer_mix == 0:
            x, target_reweighted = mixup_process(x, target_reweighted, args=None, grad=grad, noise=noise,
                                                   adv_mask1=0, adv_mask2=0, mp=mp)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)

        if layer_mix == 1:
            x, target_reweighted = mixup_process(x, target_reweighted, args=None, hidden=True)

        x = self.backbone.layer2(x)
        if layer_mix == 2:
            x, target_reweighted = mixup_process(x, target_reweighted, args=None, hidden=True)
        x = self.backbone.layer3(x)
        if layer_mix == 3:
            x, target_reweighted = mixup_process(x, target_reweighted, args=None, hidden=True)
        feats = self.backbone.layer4(x)

        x = self.pool(feats).view(x.size(0), -1)

        x = self.classifier(x)

        return x, feats, target_reweighted
