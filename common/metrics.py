import numpy as np
import torch
from numba import jit
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

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


@jit
def qwk3(a1, a2, max_rat):
    assert (len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / (1e-8 + e)
