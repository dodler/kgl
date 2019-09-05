import pandas as pd
from compress_pickle import load, dump
from tqdm import *
import cv2
import os.path as osp
import numpy as np

import os

from siim_acr_pnuemotorax.prediction_utils import fit_thresh, threshold, mask_to_rle, eval_thresh

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def custom_thresh(raw_mask):
    o = raw_mask
    o = np.flipud(o)
    o = np.rot90(o, -1)
    zz = (o > 0.5).sum()

    if zz <= 0:
        val = 0.07
        if (o > 0.15).sum() > 0:
            val = 0.09
        else:
            val = 0.03

        while (o > val).sum() <= 0 and val > 0:
            val = val - 1
    else:

        if (o > 0.85).sum() > 0:
            val = 0.31
        else:
            if (o > 0.74).sum() > 0:
                val = 0.18
            else:
                val = 0.14
    return val


vals = [
    'val_densenet169_unet_Adamw_14_bce-dice_fold0__swa_2019-08-27_17_20_16.gz',
    'val_efficientnet-b2_unet_Adamw_6_bce-dice_fold0__swa_backbone_weights_nih_pretrained_weights_effnetb2.pth_2019-08-12_11_31_27.gz',
    # 'val_densenet121_unet_Adamw_6_bce-dice_fold0__swa_2019-08-25_11_42_52.gz',
    'val_se_resnext50_32x4d_unet_Adamw_6_bce-dice_fold0__swa__backbone_weights_nih_pretrained_weights_model.pth_2019-08-04_21_27_56.gz',
    'val_dpn92_unet_Adamw_6_bce-dice_fold0__swa_2019-08-23_23_51_44.gz'
]

tests = [
    'test2_densenet169_unet_Adamw_14_bce-dice_fold0__swa_2019-08-27_17_20_16.gz',
    'test2_dpn92_unet_Adamw_6_bce-dice_fold0__swa_2019-08-23_23_51_44.gz',
    # 'test_densenet121_unet_Adamw_6_bce-dice_fold0__swa_2019-08-25_11_42_52.gz',
    'test2_efficientnet-b2_unet_Adamw_6_bce-dice_fold0__swa_backbone_weights_nih_pretrained_weights_effnetb2.pth_2019-08-12_11_31_27.gz',
    'test2_se_resnext50_32x4d_unet_Adamw_6_bce-dice_fold0__swa__backbone_weights_nih_pretrained_weights_model.pth_2019-08-04_21_27_56.gz'
]

tests = [
    "test2_densenet121_st2.gz",
    "test2_densenet169.gz",
    "test2_dpn92_st2.gz",
    "test2_efficientnet-b2_unet_Adamw_6_bce-dice_fold0_st2_hflip_no_hold_sync_bn_swa_backbone_weights_nih_pretrained_weights_effnetb2.pth_2019-09-01_10_59_29.gz",
    "test2_resnext.gz",

]

# tests=['test_efficientnet-b1_unet_Adamw_6_bce-dice_fold0_larger_holdout_swa_backbone_weights_nih_pretrained_weights_effnetb1.pth_2019-08-13_08_49_26.gz']

if __name__ == '__main__':

    # pred_path = '/var/data/siim_acr_pneumo_predicts/'
    pred_path = '/var/data/siim_stage2/'
    # blend = load(osp.join(pred_path, vals[0])).astype(np.float32) / 8.0
    # for i in range(len(vals) - 1):
    #     blend += load(osp.join(pred_path, vals[i + 1])).astype(np.float32) / 5.0
    #
    # blend /= float(len(vals))
    # #
    # dump(blend.astype(np.float16), 'val12_blend.gz')
    #
    print('loading ', tests[0])
    pred = load(osp.join(pred_path, tests[0]))
    blend_test = pred / 8.0
    print('done', pred.max())

    for i in range(len(tests) - 1):
        print('loading ', tests[i + 1])
        pred = load(osp.join(pred_path, tests[i + 1]))
        print('done', pred.max())
        blend_test += pred / 8.0

    blend_test /= len(tests)
    dump(blend_test, 'test_blend_st2.gz')
    #
    # non_empty_masks_path = '/var/ssd_1t/siim_acr_pneumo/stuff_annotations_trainval2017/annotations/masks_non_empty'
    # img_path = '/var/ssd_1t/siim_acr_pneumo/holdout/'
    # img_ids = os.listdir(img_path)
    # masks = []
    # imgs = []
    # for im in tqdm(img_ids):
    #
    #     if not osp.exists(osp.join(non_empty_masks_path, im)):
    #         continue
    #
    #     m = cv2.imread(osp.join(non_empty_masks_path, im), cv2.IMREAD_GRAYSCALE)
    #     masks.append(m)
    #
    #     img = cv2.imread(osp.join(img_path, im))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #     imgs.append(img)
    #
    # best_score, best_thresh = fit_thresh(blend, masks)
    # best_thresh = custom_thresh(blend)
    # best_score = eval_thresh(blend, masks, thresh=best_thresh)
    # print(best_score, best_thresh)
    # sub = pd.read_csv('/var/ssd_1t/siim_acr_pneumo/stage_2_sample_submission.csv')
    # test_path = '/var/ssd_1t/siim_acr_pneumo/test2'
    #
    # for i in tqdm(range(len(blend_test))):
    #     if sub.iloc[i, 1] == '-1':
    #         continue
    #     m = blend_test[i]
    #     m = threshold(m, best_thresh)
    #
    #     sub.iloc[i, 1] = mask_to_rle(m.T, 1024, 1024)
    # sub.to_csv('sub_blend.csv', index=False)
