import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def get_mel_model(mdl='b0', meta=True):
    if meta:
        if mdl == 'b0':
            return MelModel('efficientnet-b0', 1280)
        elif mdl == 'b2':
            return MelModel('efficientnet-b2', 1408)
        else:
            raise Exception('{} unsupported'.format(mdl))
    else:
        if mdl == 'b0':
            return MelModelNoMeta('efficientnet-b0', 1280)
        elif mdl == 'b2':
            return MelModelNoMeta('efficientnet-b2', 1408)
        else:
            raise Exception('{} unsupported'.format(mdl))


class MelModel(torch.nn.Module):
    def __init__(self, name, n_last=1280):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(name)
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_last, out_channels=n_last, kernel_size=7, groups=n_last),
            nn.ELU(),
            nn.BatchNorm2d(n_last),
        )

        self.meta_head = nn.Sequential(nn.Linear(12, 500),
                                       nn.BatchNorm1d(500),
                                       nn.ELU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(500, 250),  # FC layer output will have 250 features
                                       nn.BatchNorm1d(250),
                                       nn.ELU(),
                                       nn.Dropout(p=0.2))
        self.last_head = nn.Sequential(
            nn.Linear(n_last + 250, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.BatchNorm1d(1),
            nn.ELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x, meta = x
        cnn_features = self.backbone.extract_features(x)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, output_size=7)
        cnn_features = self.last_conv(cnn_features).squeeze()
        meta_features = self.meta_head(meta)
        x = torch.cat([cnn_features, meta_features], dim=1)

        return self.last_head(x).squeeze()


class MelModelNoMeta(torch.nn.Module):
    def __init__(self, name, n_last=1280):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(name)
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_last, out_channels=n_last, kernel_size=7, groups=n_last),
            nn.ELU(),
            nn.BatchNorm2d(n_last),
        )

        self.last_head = nn.Sequential(
            nn.Linear(n_last, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.BatchNorm1d(1),
            nn.ELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x, meta = x
        cnn_features = self.backbone.extract_features(x)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, output_size=7)
        cnn_features = self.last_conv(cnn_features).squeeze()

        return self.last_head(cnn_features).squeeze()
