import os
import torch.nn as nn
from tqdm import *
import numpy as np
from numba import njit, jit


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


class DiceMetric(nn.Module):
    __name__ = 'dice'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold
        self.beta = beta

    def forward(self, pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1).float()
        m2 = target.view(num, -1).float()

        m1 = (m1>0.5).float()*1
        m2 = (m2>0.5).float()*1

        intersection = (m1 * m2).sum().float()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dice(seg, gt, k=1):
    return np.sum(seg[gt == k] == k) * 2.0 / (np.sum(seg[seg == k] == k) + np.sum(gt[gt == k] == k))


def rle2mask(rle, width=1024, height=1024):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def mask_to_rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
    return " " + " ".join(rle)


def eval_thresh(preds, gt_masks, thresh=0.5):
    result = np.zeros(preds.shape[0])
    for i in np.arange(len(preds)):
        result[i] = dice(threshold(preds[i], thresh=thresh), gt_masks[i])
    return result.mean()


def threshold(m, thresh=0.5):
    t = m.copy()
    t[t > thresh] = 1
    t[t <= thresh] = 0
    return t.astype(np.uint8)


def fit_thresh(preds, masks):
    left = 0
    right = 1
    score_delta = 1
    delta_thresh = 0.00001
    n = 20
    best_score = 0
    while score_delta > delta_thresh:
        print('iterating, left=', left, 'right=', right, 'score_delta=', score_delta, 'best score', best_score)
        m = np.zeros(n)
        x = np.linspace(left, right, n)
        for i in tqdm(range(n)):
            r = eval_thresh(preds, masks, thresh=x[i])
            m[i] = r

        score = np.max(m)
        best_thresh = x[np.argmax(m)]
        score_delta = score - best_score
        if score_delta > delta_thresh:
            best_score = score
            left += (best_thresh - left) / 2.0
            right -= (right - best_thresh) / 2.0
        if left > right:
            break
    return np.max(m), x[np.argmax(m)]


sigmoid(np.zeros(10))
