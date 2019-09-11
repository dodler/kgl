#!/usr/bin/env python
"""Test UNet and create a Kaggle submission."""

from segmentation.segmentation import load_seg_model
from segmentation.segmentation import aug_resize
from segmentation.segmentation import toRunLength
from segmentation.segmentation import gzip_save
from siim_acr_pnuemotorax.siim_data import SIIMDatasetSegmentation

__author__ = 'Erdene-Ochir Tuguldur, Yuan Xu'

import time
import argparse
from tqdm import tqdm

import pandas as pd

import torch
from torch.utils.data import DataLoader
import numpy as np


def predict(model, batch, flipped_batch, use_gpu):
    inputs = batch
    if use_gpu:
        inputs = inputs.cuda()
    outputs, _, _ = model(inputs)
    # probs = torch.sigmoid(outputs)
    probs = outputs.squeeze(1).cpu().numpy()
    return probs


def test():
    test_dataset = SIIMDatasetSegmentation(image_dir='/var/ssd_1t/siim_acr_pneumo/test_png',
                                            mask_dir=None,
                                            aug=aug_resize)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.dataload_workers_nums, shuffle=False)

    model.eval()
    torch.set_grad_enabled(False)

    prediction = {}
    submission = {}
    pbar = tqdm(test_dataloader, unit="images", unit_scale=test_dataloader.batch_size, disable=None)

    raw_predictions=[]

    empty_images_count = 0
    for batch in pbar:
        probs = predict(model, batch, None, use_gpu=use_gpu)
        raw_predictions.append(probs)
        pred = probs > args.threshold
        empty_images_count += (pred.sum(axis=(1, 2)) == 0).sum()

        probs_uint16 = (65535 * probs).astype(dtype=np.uint16)

        # image_ids = batch['image_id']
        # prediction.update(dict(zip(image_ids, probs_uint16)))
        # rle = toRunLength(pred)
        # submission.update(dict(zip(image_ids, rle)))

    empty_images_percentage = empty_images_count / len(prediction)
    print("empty images: %.2f%% (in public LB 38%%)" % (100 * empty_images_percentage))

    import pickle
    pickle.dump(raw_predictions, open('raw_pred.pkl','wb'))

    gzip_save('-'.join([args.output_prefix, 'probabilities.pkl.gz']), prediction)
    # sub = pd.DataFrame.from_dict(submission, orient='index')
    # sub.index.names = ['id']
    # sub.columns = ['rle_mask']
    # sub.to_csv('-'.join([args.output_prefix, 'submission.csv']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=1, help='batch size')
    parser.add_argument("--dataload-workers-nums", type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--tta', action='store_true', help='test time augmentation')
    parser.add_argument('--seed', type=int, default=None, help='manual seed for deterministic')
    parser.add_argument("--threshold", type=float, default=0.5, help='probability threshold')
    parser.add_argument("--output-prefix", type=str, default='noprefix', help='prefix string for output files')
    parser.add_argument('--resize', action='store_true', help='resize to 128x128 instead of reflective padding')
    parser.add_argument("model", help='a pretrained neural network model')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print('use_gpu', use_gpu)

    print("loading model...")
    model = load_seg_model(args.model)
    model.float()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
    model.cuda()

    print("testing %s..." % args.model)
    since = time.time()
    test()
    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                     time_elapsed % 60)
    print("finished")