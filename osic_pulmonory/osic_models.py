import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


def get_osic_model(model):
    return OsicEffNet(n_tab=4)


class OsicEffNet(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
        self.n_tab = n_tab

        self.back = EfficientNet.from_pretrained('efficientnet-b2', in_channels=1)

        n_last = 1408
        n_meta_intermediate = 256

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=n_last, out_channels=n_last, groups=n_last, kernel_size=7),
            nn.BatchNorm2d(n_last),
            nn.ELU(),
        )

        self.meta_head = nn.Sequential(nn.Linear(self.n_tab, n_meta_intermediate),
                                       nn.BatchNorm1d(n_meta_intermediate),
                                       nn.ELU(),
                                       nn.Dropout(p=0.2))

        self.last_head = nn.Sequential(
            nn.Linear(n_last + n_meta_intermediate, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.BatchNorm1d(1),
            nn.ELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        img, tab = x
        b = img.shape[0]
        img_features = self.back.extract_features(img)
        img_features = F.adaptive_avg_pool2d(img_features, 7)
        img_features = self.last_conv(img_features).reshape(b, -1)

        tab_features = self.meta_head(tab)
        img = torch.cat([img_features, tab_features], dim=1)

        return self.last_head(img).squeeze()


if __name__ == '__main__':
    model = OsicEffNet(n_tab=10)
    model.eval()
    out = model([torch.randn(1, 1, 224, 224), torch.randn(1, 10)])
    print(out)
