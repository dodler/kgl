from typing import Dict

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from timm.models.layers import create_conv2d
from torch import nn
from torchvision.models import resnet18, resnet50, resnet34, resnet101
import timm

from lyft_motion_prediction import pytorch_neg_multi_log_likelihood_batch


class LyftModel(nn.Module):

    def __init__(self, cfg: Dict):
        super().__init__()

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone = EfficientNet.from_pretrained('efficientnet-b4', in_channels=num_in_channels)

        backbone_out_features = 1792

        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        b, c, d, d = x.shape
        x = F.adaptive_avg_pool2d(x, 1).squeeze()

        x = self.head(x)
        x = self.logit(x)

        return x


class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        timm_models = ['tf_efficientnet_l2_ns_475', 'resnest200e', 'resnest101e', 'swsl_resnext101_32x8d']

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.model_arch = cfg['model_params']['model_architecture']

        if cfg['model_params']['model_architecture'] == 'resnet18':
            backbone = resnet18(pretrained=True, progress=True)
            backbone_out_features = 512
        elif cfg['model_params']['model_architecture'] == 'resnet50':
            backbone = resnet50(pretrained=True, progress=True)
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnet34':
            backbone = resnet34(pretrained=True, progress=True)
            backbone_out_features = 512
        elif cfg['model_params']['model_architecture'] == 'resnet101':
            backbone = resnet101(pretrained=True, progress=True)
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnext50_32x4d':
            from torchvision.models import resnext50_32x4d
            backbone = resnext50_32x4d(pretrained=True, progress=True)
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnext101_32x8d':
            from torchvision.models import resnext101_32x8d
            backbone = resnext101_32x8d(pretrained=True, progress=True)
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnext101_32x16d_wsl':
            backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnext101_32x32d_wsl':
            backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x32d_wsl')
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnext101_32x48d_wsl':
            backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'tf_efficientnet_l2_ns_475':
            backbone = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True)
            backbone_out_features = 5504
        elif cfg['model_params']['model_architecture'] == 'resnest200e':
            backbone = timm.create_model('resnest200e', pretrained=True, in_chans=num_in_channels)
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'resnest101e':
            backbone = timm.create_model('resnest101e', pretrained=True, in_chans=num_in_channels)
            backbone_out_features = 2048
        elif cfg['model_params']['model_architecture'] == 'swsl_resnext101_32x8d':
            backbone = timm.create_model('swsl_resnext101_32x8d', pretrained=True, in_chans=num_in_channels)
            backbone_out_features = 2048
        else:
            raise Exception('not supported {}'.format(cfg['model_params']['model_architecture']))

        self.backbone = backbone

        if self.model_arch not in timm_models:
            self.backbone.conv1 = nn.Conv2d(
                num_in_channels,
                self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=False,
            )

        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        if cfg['model_params']['head'] == 'v0':
            self.head = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(in_features=backbone_out_features, out_features=4096),
            )
            self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)
        elif cfg['model_params']['head'] == 'v1':
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features=backbone_out_features, out_features=4096),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=4096, out_features=2048),
                nn.LeakyReLU(),
            )
            self.logit = nn.Linear(2048, out_features=self.num_preds + num_modes)
        else:
            raise Exception('head {} not supported'.format(cfg['model_params']['head']))

    def forward(self, x):
        if self.model_arch in ['tf_efficientnet_l2_ns_475', 'efficientnet_b3', 'resnest200e', 'resnest101e', 'swsl_resnext101_32x8d']:
            x = self.backbone.forward_features(x)
            x = F.adaptive_avg_pool2d(x, output_size=1)
        else:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


class LyftMultiRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun

    def forward(self, image, targets, target_availabilities):
        pred, confidences = self.predictor(image)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()
        }
        return loss, metrics


if __name__ == '__main__':
    cfg = {'model_params':
        {
            'model_architecture': 'tf_efficientnet_l2_ns_475'
        }
    }

# ['adv_inception_v3', 'cspdarknet53', 'cspresnet50', 'cspresnext50',
#  'densenet121', 'densenet161', 'densenet169', 'densenet201',
#  'densenetblur121d', 'dla34', 'dla46_c', 'dla46x_c', 'dla60', 'dla60_res2net',
#  'dla60_res2next', 'dla60x', 'dla60x_c', 'dla102', 'dla102x', 'dla102x2', 'dla169',
#  'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'ecaresnet50d', 'ecaresnet50d_pruned',
#  'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnetlight',
#  'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b1_pruned',
#  'efficientnet_b2', 'efficientnet_b2_pruned', 'efficientnet_b2a', 'efficientnet_b3',
#  'efficientnet_b3_pruned', 'efficientnet_b3a', 'efficientnet_em', 'efficientnet_es',
#  'efficientnet_lite0', 'ens_adv_inception_resnet_v2', 'ese_vovnet19b_dw', 'ese_vovnet39b',
#  'fbnetc_100', 'gluon_inception_v3', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b', 'gluon_resnet50_v1b',
#  'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 'gluon_resnet101_v1b',
#  'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 'gluon_resnet152_v1b',
#  'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 'gluon_resnext50_32x4d',
#  'gluon_resnext101_32x4d', 'gluon_resnext101_64x4d', 'gluon_senet154', 'gluon_seresnext50_32x4d',
#  'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'gluon_xception65', 'hrnet_w18',
#  'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40',
#  'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 'ig_resnext101_32x8d', 'ig_resnext101_32x16d',
#  'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'inception_resnet_v2', 'inception_v3',
#  'inception_v4', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50',
#  'legacy_seresnet101', 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d',
#  'legacy_seresnext101_32x4d', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl',
#  'mnasnet_100', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d',
#  'mobilenetv2_140', 'mobilenetv3_large_100', 'mobilenetv3_rw', 'nasnetalarge',
#  'pnasnet5large', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008',
#  'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080',
#  'regnetx_120', 'regnetx_160', 'regnetx_320', 'regnety_002', 'regnety_004',
#  'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040',
#  'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320',
#  'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s',
#  'res2net50_48w_2s', 'res2net101_26w_4s', 'res2next50', 'resnest14d', 'resnest26d',
#  'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 'resnest101e', 'resnest200e',
#  'resnest269e', 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet34', 'resnet34d',
#  'resnet50', 'resnet50d', 'resnetblur50', 'resnext50_32x4d', 'resnext50d_32x4d',
#  'resnext101_32x8d', 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200',
#  'selecsls42b', 'selecsls60', 'selecsls60b', 'semnasnet_100', 'seresnet50',
#  'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26tn_32x4d', 'seresnext50_32x4d',
#  'skresnet18', 'skresnet34', 'skresnext50_32x4d', 'spnasnet_100',
#  'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d',
#  'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 'swsl_resnet18', 'swsl_resnet50',
#  'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d',
#  'swsl_resnext101_32x16d', 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap',
#  'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 'tf_efficientnet_b1_ap',
#  'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap',
#  'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap',
#  'tf_efficientnet_b3_ns', 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap',
#  'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 'tf_efficientnet_b5_ap',
#  'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap',
#  'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap',
#  'tf_efficientnet_b7_ns', 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap',
#  'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e',
#  'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em',
#  'tf_efficientnet_es', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475',
#  'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2',
#  'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_inception_v3',
#  'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s', 'tf_mobilenetv3_large_075',
#  'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075',
#  'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 'tresnet_l', 'tresnet_l_448',
#  'tresnet_m', 'tresnet_m_448', 'tresnet_xl', 'tresnet_xl_448', 'tv_densenet121', 'tv_resnet34',
#  'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 'vit_base_patch16_224',
#  'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_224', 'vit_large_patch16_384',
#  'vit_large_patch32_384', 'vit_small_patch16_224', 'wide_resnet50_2', 'wide_resnet101_2', 'xception',
#  'xception41', 'xception65', 'xception71']
