from torch.nn.modules.loss import _Loss

import math
import torch
import torch.nn.functional as F

__all__ = ['sigmoid_focal_loss', 'soft_jaccard_score', 'soft_dice_score', 'wing_loss']


def sigmoid_focal_loss(input: torch.Tensor,
                       target: torch.Tensor,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
    References::
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * target + (1 - alpha) * (1 - target))

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss


def reduced_focal_loss(input: torch.Tensor,
                       target: torch.Tensor,
                       threshold=0.5,
                       gamma=2.0,
                       reduction='mean'):
    """Compute reduced focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
    References::
        https://arxiv.org/abs/1903.01347
    """
    target = target.type(input.type())

    logpt = -F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(logpt)

    # compute the loss
    focal_reduction = ((1. - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = - focal_reduction * logpt

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss


def soft_jaccard_score(pred: torch.Tensor,
                       target: torch.Tensor,
                       smooth=1e-3,
                       from_logits=False) -> torch.Tensor:
    if from_logits:
        pred = pred.sigmoid()

    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    iou = intersection / (union - intersection + smooth)
    return iou


def soft_dice_score(pred: torch.Tensor,
                    target: torch.Tensor,
                    smooth=1e-3,
                    from_logits=False) -> torch.Tensor:
    if from_logits:
        pred = pred.sigmoid()

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) + smooth
    return 2 * intersection / union


def wing_loss(prediction: torch.Tensor, target: torch.Tensor, width=5, curvature=0.5, reduction='mean'):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param prediction:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - prediction).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == 'sum':
        loss = loss.sum()

    if reduction == 'mean':
        loss = loss.mean()

    return loss



class BinaryFocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None, reduction='mean', reduced=False, threshold=0.5):
        """
        :param alpha:
        :param gamma:
        :param ignore_index:
        :param reduced:
        :param threshold:
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        if reduced:
            self.focal_loss = partial(reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.focal_loss = partial(sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, label_input, label_target):
        """Compute focal loss for binary classification problem.
        """
        label_target = label_target.view(-1)
        label_input = label_input.view(-1)

        if self.ignore_index is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = label_target != self.ignore_index
            label_input = label_input[not_ignored]
            label_target = label_target[not_ignored]

        loss = self.focal_loss(label_input, label_target)
        return loss


class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        """
        Focal loss for multi-class problem.
        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += sigmoid_focal_loss(cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha)
        return loss