import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

from cassava.repvgg import get_RepVGG_func_by_name


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

state_dicts={
    'RepVGG-B2g4': '/home/smith/RepVGG-B2g4-200epochs-train.pth',
    'RepVGG-B0': '/home/smith/RepVGG-B0-train.pth',
    'RepVGG-B1': '/home/smith/RepVGG-B1-train.pth',
}


class CassavaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if 'pool' not in cfg:
            cfg['pool'] = 'avg_pool'

        self.backbone_name = cfg['backbone']

        if 'RepVGG' in cfg['backbone']:
            repvgg_build_func = get_RepVGG_func_by_name(cfg['backbone'])
            self.backbone = repvgg_build_func(deploy=False)
            ckpt = state_dicts[cfg['backbone']]
            ckpt=torch.load(ckpt, map_location='cpu')
            self.backbone.load_state_dict(ckpt)
        elif 'deit' in cfg['backbone']:
            self.backbone = torch.hub.load('facebookresearch/deit:main', cfg['backbone'], pretrained=True)
        else:
            self.backbone = timm.create_model(cfg['backbone'], pretrained=True)
        n_hidden = 4096
        if 'efficient' in cfg['backbone']:
            n_out = self.backbone.classifier.in_features
        elif 'deit' in cfg['backbone']:
            n_out = self.backbone.head.in_features
        elif 'inception' in cfg['backbone']:
            n_out = self.backbone.last_linear.in_features
        elif 'hrnet' in cfg['backbone']:
            n_out = self.backbone.classifier.in_features
        elif 'RepVGG' in cfg['backbone']:
            n_out = self.backbone.linear.in_features
        elif 'vit' in cfg['backbone']:
            n_out = self.backbone.head.in_features
        else:
            n_out = self.backbone.fc.in_features
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
        if 'deit' not in self.backbone_name and 'vit' not in self.backbone_name:
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
