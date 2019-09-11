from segmentation_models_pytorch.base.model import Model
from segmentation_models_pytorch.encoders import get_encoder

import torch.nn as nn
import torch
from segmentation.effnet_decoder import EfficientNetEncoder
from segmentation.oc_unet_decoder import OCUnetDecoder
from segmentation.unet_decoder import UnetDecoder


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation):
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid" or "softmax"')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            x = self.activation(x)

        return x


class Unet(EncoderDecoder):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: one of [``sigmoid``, ``softmax``, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            use_oc_module=False,
    ):
        if 'efficientnet' in encoder_name:
            encoder = EfficientNetEncoder.from_pretrained(encoder_name)
        else:
            encoder = get_encoder(
                encoder_name,
                encoder_weights=encoder_weights
            )

        if use_oc_module:
            decoder = OCUnetDecoder(
                encoder_channels=encoder.out_shapes,
                decoder_channels=decoder_channels,
                final_channels=classes,
                use_batchnorm=decoder_use_batchnorm,
                center=center,
            )

        else:
            decoder = UnetDecoder(
                encoder_channels=encoder.out_shapes,
                decoder_channels=decoder_channels,
                final_channels=classes,
                use_batchnorm=decoder_use_batchnorm,
                center=center,
            )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)


if __name__ == '__main__':
    ACTIVATION = 'sigmoid'
    model = Unet(encoder_name='efficientnet-b5', encoder_weights='imagenet', classes=1, activation=ACTIVATION)
    # model = Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=1, activation=ACTIVATION)
    model(torch.zeros(1, 3, 1024, 1024))
