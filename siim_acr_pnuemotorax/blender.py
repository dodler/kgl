import pandas as pd
from compress_pickle import load
from tqdm import *
import cv2
import os.path as osp
import numpy as np

import os

from siim_acr_pnuemotorax.prediction_utils import fit_thresh, threshold, mask_to_rle

os.environ['CUDA_VISIBLE_DEVICES'] = ''

if __name__ == '__main__':


    pred_path = '/var/data/siim_acr_pneumo_predicts/'
    blend = load(osp.join(pred_path,'val_preds_blend_5_fold_effnet_b0_unet_swa_test_fp16.pkl.gz')).astype(np.float32) / 5.0 +\
            load(osp.join(pred_path,'val_preds_blend_5_fold_effnet_b1_unet_swa_test_fp16.pkl.gz')).astype(np.float32) / 5.0 + \
            load(osp.join(pred_path, 'val_preds_blend_5_fold_se_resnext50_32x4d_unet_Adamw_6_bce-dice_fold0__swa__backbone_weights_nih_pretrained_weights_model_fp16.pkl.gz')).astype(np.float32)/5.0

    blend /= 3.0

    blend_test=load(osp.join(pred_path, 'raw_pred_blend_fold_effnet_b0_unet_swa_test_fp16.pkl.gz')).astype(np.float32)/5.0+\
               load(osp.join(pred_path, 'raw_pred_blend_5_fold_se_resnext50_32x4d_unet_Adamw_6_bce-dice_fold0__swa__backbone_weights_nih_pretrained_weights_model_fp16.pkl.gz')).astype(np.float32)+\
               load(osp.join(pred_path, 'raw_pred_blend_fold_effnet_b1_unet_swa_test_fp16.pkl.gz')).astype(np.float32)/5.0

    blend_test /= 3.0

    non_empty_masks_path = '/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty'
    img_path = '/var/ssd_1t/siim_acr_pneumo/val2017'
    img_ids = os.listdir(img_path)
    masks = []
    imgs = []
    for im in tqdm(img_ids):

        if not osp.exists(osp.join(non_empty_masks_path, im)):
            continue

        m = cv2.imread(osp.join(non_empty_masks_path, im), cv2.IMREAD_GRAYSCALE)
        masks.append(m)

        img = cv2.imread(osp.join(img_path, im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgs.append(img)

    best_score, best_thresh = fit_thresh(blend, masks)
    print(best_score, best_thresh)
    sub = pd.read_csv('sub8810.csv')
    test_path = '/var/ssd_1t/siim_acr_pneumo/test_png'

    for i in tqdm(range(len(blend_test))):
        if sub.iloc[i, 1] == '-1':
            continue
        m = blend_test[i]
        m = threshold(m, best_thresh)

        sub.iloc[i, 1] = mask_to_rle(m.T, 1024, 1024)
    sub.to_csv('sub_blend.csv', index=False)
