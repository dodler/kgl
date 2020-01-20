import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet_params, get_same_padding_conv2d
import torch.nn as nn
import torch.nn.functional as F


class BengEffNetClassifier(nn.Module):
    def forward(self, x):
        if self.input_bn:
            x = self.bn_in(x)
        x = self.net.extract_features(x)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.net._dropout:
            x = F.dropout(x, p=self.net._dropout, training=self.training)

        return self.cls1(x), self.cls2(x), self.cls3(x)

    def __init__(self, name='efficientnet-b0', pretrained=True, input_bn=True):
        super().__init__()
        self.input_bn = input_bn
        self.name = name
        if pretrained:
            self.net = EfficientNet.from_pretrained(model_name=name)
        else:
            self.net = EfficientNet.from_name(model_name=name)

        params = efficientnet_params(model_name=name)
        Conv2d = get_same_padding_conv2d(image_size=params[2])
        conv_stem_filts = {
            'efficientnet-b0': 32,
            'efficientnet-b4': 48,
            'efficientnet-b7': 64,
        }

        linear_size = {
            'efficientnet-b0': 1280,
            'efficientnet-b4': 1792,
            'efficientnet-b7': 2560
        }
        self.net._conv_stem = Conv2d(1, conv_stem_filts[name], kernel_size=(3, 3), stride=(2, 2), bias=False)

        self.cls1 = nn.Linear(linear_size[name], 168)
        self.cls2 = nn.Linear(linear_size[name], 11)
        self.cls3 = nn.Linear(linear_size[name], 7)

        if input_bn:
            self.bn_in = nn.BatchNorm2d(1)


if __name__ == '__main__':
    net = BengEffNetClassifier(name='efficientnet-b7')
    print(net.net)

    inp = torch.randn(2, 1, 224, 224)
    x1, x2, x3 = net(inp)
    print(x1.shape, x2.shape, x3.shape)
