import os

from tqdm import *
import numpy as np
from numba import njit, jit


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
