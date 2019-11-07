import torch
from efficientnet_pytorch import EfficientNet
from pretrainedmodels import inceptionresnetv2, inceptionv4, inceptionv3
from torch.nn import Conv2d, Sequential, BatchNorm2d
from torchvision.models.inception import BasicConv2d


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_model(model_name='resnext101_32x16d_wsl', n_classes=6, raw=False):

    model = None

    if model_name.startswith('eff'):
        model = EfficientNet.from_pretrained(model_name)
        last_output_size=2048

        if model_name.endswith('b1') or model_name.endswith('b0'):
            last_output_size = 1280
            if raw:
                raise Exception('Checkout for other effnet types the channel size out last output')
        elif model_name.endswith('b2'):
            if raw:
                model._conv_stem = Conv2d(1, 32, kernel_size=3, stride=2, bias=False)
                # torch.nn.init.xavier_normal_(model._conv_stem.weight)
            last_output_size = 1408
        elif model_name.endswith('b4'):
            if raw:
                model._conv_stem = Conv2d(1, 48, kernel_size=3, stride=2, bias=False, padding=(0,1))
                # torch.nn.init.xavier_normal_(model._conv_stem.weight)
            last_output_size = 1792
        else:
            raise Exception('Checkout for other effnet types the channel size out last output')

        model._fc = torch.nn.Linear(last_output_size, n_classes)

    elif model_name.startswith('resnext101_32x') and model_name.endswith('wsl'):
        model = torch.hub.load('facebookresearch/WSL-Images', model_name)
        if raw:
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(2048, n_classes)
    elif model_name == 'inceptionv3':
        model = inceptionv3()
        model.last_linear = torch.nn.Linear(2048, n_classes)
        if raw:
            model.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)

    if model is None:
        raise Exception('failed to instantiate model: '+ model_name)

    return model


