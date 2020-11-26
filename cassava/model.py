import torch.nn as nn
import timm
import torch.nn.functional as F


class CassavaModel(nn.Module):
    def __init__(self, backbone='resnet18'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True)
        n_hidden = 4096
        n_out = 512
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_out, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ELU(),
            nn.Linear(n_hidden, 5),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        return self.head(x)
