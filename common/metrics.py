import torch
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np

NO_VAL = [-1]


def catalyst_roc_auc(pred, gt, *args, **kwargs):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy().astype(np.uint8)
    try:
        return [roc_auc_score(gt.reshape(-1), pred.reshape(-1))]
    except Exception as e:
        return NO_VAL


def catalyst_acc_score(pred, gt, *args, **kwargs):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)
    gt = gt.detach().cpu().numpy().astype(np.uint8)
    try:
        return [accuracy_score(gt.reshape(-1), pred.reshape(-1))]
    except Exception as e:
        return NO_VAL


def catalyst_logloss(pred, gt, *args, **kwargs):
    pred = torch.sigmoid(pred).detach().cpu().numpy()
    gt = gt.detach().cpu().numpy().astype(np.uint8)
    try:
        return [log_loss(gt.reshape(-1), pred.reshape(-1))]
    except Exception as e:
        return NO_VAL
