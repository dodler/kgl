import torch
from efficientnet_pytorch import EfficientNet
import pretrainedmodels as pm


def get_model(name):
    if name == 'effb3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model._fc = torch.nn.Linear(1536, 2)
        return model

    if name == 'effb1':
        model = EfficientNet.from_pretrained('efficientnet-b1')
        model._fc = torch.nn.Linear(1280, 2)
        return model

    if name == 'dnet161':
        model = pm.densenet161()
        model.last_linear = torch.nn.Linear(2048, 2)
        return model

    if name == 'effb0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = torch.nn.Linear(1280, 2)
        return model

    if name == 'dnet121':
        model = pm.densenet121(pretrained='imagenet')
        model.last_linear = torch.nn.Linear(1024, 2)
        return model

    if name == 'se_resnext50_32x4d':
        model = pm.se_resnext50_32x4d()
        model.last_linear = torch.nn.Linear(2048, 2)
        return model

    if name == 'effb7':
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model._fc = torch.nn.Linear(2048, 2)
        return model

    raise Exception('model {} is not supported'.format(name))
