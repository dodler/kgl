from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch.nn as nn


class RsnaStrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')

        self.last_channel = 1280
        self.cls = nn.Sequential(
            nn.Linear(self.last_channel, 10)
        )

    def forward(self, x):
        b = x.shape[0]
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(b, -1)
        return self.cls(x).squeeze()

