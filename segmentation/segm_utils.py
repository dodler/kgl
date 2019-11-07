import numpy as np
from numba import jit, njit


def threshold(m, thresh=0.5):
    return (m > thresh).astype(np.uint8)


@jit
def threshold_jit(t, thresh=0.5):
    return (t > thresh).astype(np.uint8)


@njit
def threshold_njit(t, thresh=0.5):
    return (t > thresh).astype(np.uint8)


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


beta = 1
eps = 1e-7


def dice(pred, target):
    smooth = 1.
    num = pred.shape[0]
    m1 = pred.reshape(-1).astype(np.float32)
    m2 = target.reshape(-1).astype(np.float32)

    m1 = (m1 > 0.5).astype(np.float32) * 1
    m2 = (m2 > 0.5).astype(np.float32) * 1

    intersection = (m1 * m2).sum().astype(np.float32)

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


@jit
def dice_jit(pred, target):
    smooth = 1.
    num = pred.shape[0]
    m1 = pred.reshape(num, -1) * 1.0
    m2 = target.reshape(num, -1) * 1.0
    intersection = np.sum(m1 * m2).astype(np.float32)

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


@njit
def dice_njit(pred, target):
    smooth = 1.
    num = pred.shape[0]
    m1 = pred.reshape(num, -1) * 1.0
    m2 = target.reshape(num, -1) * 1.0
    intersection = np.sum(m1 * m2) * 1.0

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def fit_thresh_single_old(pred, gt):
    ths=np.linspace(0,1,100)
    metrics = []
    for th in ths:
        metrics.append(dice(threshold(pred, th), gt))
    return np.array(metrics)

@jit
def fit_thresh_single_jit(pred, gt):
    ths=np.linspace(0,1,100)
    metrics = []
    for th in ths:
        metrics.append(dice(threshold(pred, th), gt))
    return np.array(metrics)