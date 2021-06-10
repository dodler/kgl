import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm)

    return output


class CurricularFace(nn.Module):
    def __init__(self, embedding_size, num_classes, m=0.5, s=64.):
        super(CurricularFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = num_classes
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(embedding_size, num_classes))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, x, y):
        x = l2_norm(x, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(x, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, x.size(0)), y].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, y.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y=None):
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if y is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / x.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


class ArcFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y=None):
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if y is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class SphereFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=1.35):
        super(SphereFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y=None):
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if y is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class CosFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y=None):
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if y is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class MV_Arc(Module):
    def __init__(self, fc_type='MV-AM', margin=0.35, t=0.2, scale=32, embedding_size=512, num_class=72690,
                 easy_margin=True):
        super(FC, self).__init__()
        self.weight = Parameter(torch.Tensor(embedding_size, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.t = t
        self.easy_margin = easy_margin
        self.scale = scale
        self.fc_type = fc_type
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # duplication formula
        self.iter = 0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x, y=None):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)
        if y is None:
            return cos_theta
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        batch_size = y.size(0)
        gt = cos_theta[torch.arange(0, batch_size), y].view(-1, 1)  # ground truth score

        if self.fc_type == 'MV-AM':
            mask = cos_theta > gt - self.margin
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin
        elif self.fc_type == 'MV-Arc':
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)
        else:
            raise Exception('unknown fc type!')

        cos_theta.scatter_(1, y.data.view(-1, 1), final_gt)
        cos_theta *= self.scale
        return cos_theta