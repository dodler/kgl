import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class CassavaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if 'pool' not in cfg:
            cfg['pool'] = 'avg_pool'
        self.backbone = timm.create_model(cfg['backbone'], pretrained=True)
        n_hidden = 4096
        n_out = cfg['n_out']
        self.n_out = n_out
        head = cfg['head']
        if head == 'v0':
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(n_out, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ELU(),
                nn.Linear(n_hidden, 5),
            )
        elif head == 'v1':
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_out, 5))
        else:
            raise Exception('unsupported {}'.format(head))

        if cfg['pool'] == 'gem':
            self.pool = GeM()
        elif cfg['pool'] == 'avg_pool':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif cfg['pool'] == 'avg_pool+conv':
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(7),
                nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=7, bias=False)
            )
        else:
            raise Exception('pool not supported {}'.format(cfg['pool']))

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x).squeeze()
        return self.head(x)


if __name__ == '__main__':
    gem = GeM()
    x = torch.randn(1, 512, 14, 14)
    print(gem(x).shape)

    pool = nn.Sequential(
        nn.AdaptiveAvgPool2d(7),
        nn.Conv2d(kernel_size=7, bias=False, in_channels=512, out_channels=512)
    )
    print(pool(x).shape)
