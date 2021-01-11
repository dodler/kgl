import timm
import torch.nn as nn


class RanzcrModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(cfg['backbone'], pretrained=True)
        if 'efficient' in cfg['backbone']:
            n_out = self.backbone.classifier.in_features
        else:
            n_out = self.backbone.fc.in_features

        n_classes = 11
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_out, n_classes))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x).squeeze()
        return self.head(x)
