import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base.model import Model
from segmentation_models_pytorch.common.blocks import Conv2dReLU

from segmentation.os_net import BaseOC


class OCDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, use_self_attention=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            BaseOC(in_channels=out_channels,
                   out_channels=out_channels,
                   key_channels=out_channels // 2,
                   value_channels=out_channels // 2,
                   dropout=0.05,
                   use_self_attention=use_self_attention)
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class CenterBlock(OCDecoderBlock):

    def forward(self, x):
        return self.block(x)


class OCUnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = OCDecoderBlock(in_channels[0], out_channels[0],
                                     use_batchnorm=use_batchnorm, use_self_attention=True)
        self.layer2 = OCDecoderBlock(in_channels[1], out_channels[1],
                                     use_batchnorm=use_batchnorm, use_self_attention=True)
        self.layer3 = OCDecoderBlock(in_channels[2], out_channels[2],
                                     use_batchnorm=use_batchnorm, use_self_attention=True)
        self.layer4 = OCDecoderBlock(in_channels[3], out_channels[3],
                                     use_batchnorm=use_batchnorm, use_self_attention=False)
        self.layer5 = OCDecoderBlock(in_channels[4], out_channels[4],
                                     use_batchnorm=use_batchnorm, use_self_attention=False)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
