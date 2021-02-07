import timm
import torch.nn as nn

from cassava.pazzle_mix.pazzle_mixup import mixup_process, to_one_hot, get_lambda


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

    def forward(self, x, y=None, grad=None):
        # works only for efn

        if y is not None:
            target_reweighted = to_one_hot(y, 5)
        else:
            target_reweighted = None

        if y is not None:
            x, target_reweighted = mixup_process(x, target_reweighted, args=None, grad=grad, noise=None,
                                                 adv_mask1=0, adv_mask2=0, mp=4)

        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)

        # if y is not None:
        #     lam = get_lambda(alpha=2.0)  # from github
        #     x, target_reweighted = mixup_process(out=x, target_reweighted=target_reweighted, lam=lam)

        x = self.backbone.blocks(x)

        # if y is not None:
        #     lam = get_lambda(alpha=2.0)  # from github
        #     x, target_reweighted = mixup_process(out=x, target_reweighted=target_reweighted, lam=lam)

        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        feats = self.backbone.act2(x)

        x = self.pool(feats).view(x.size(0), -1)

        x = self.classifier(x)

        return x, feats, target_reweighted
