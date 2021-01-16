import torch.nn as nn
import timm
import torch.nn.functional as F


class CassavaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = timm.create_model(cfg['backbone'], pretrained=True)
        if 'efficient' in cfg['backbone']:
            n_features = backbone.classifier.in_features
        else:
            n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats