import pandas as pd
import torch
from efficientnet_pytorch import EfficientNet

from severstal.sev_data import SegData, SevClass


def load_pretrained_weights(path, model):
    if path is None or len(path) == 0:
        return
    pretrained_dict = torch.load(path, map_location='cpu')

    model_dict = model.encoder.state_dict()
    pretrained_dict = {k.replace('feature_extr.', ''): v for k, v in pretrained_dict.items()
                       if k.replace('feature_extr.', '') in model_dict}
    model_dict.update(pretrained_dict)

    model_dict.pop('_fc.weight')
    model_dict.pop('_fc.bias')

    if 'se_resne' in path:
        model_dict['last_linear.bias'] = None
        model_dict['last_linear.weight'] = None
    print('loading pretrained weights')
    model.encoder.load_state_dict(model_dict, strict=False)
    print('pretrained weights load success')


def seg_from_folds(image_dir,
                   mask_dir,
                   aug_trn,
                   aug_val,
                   folds_path='/home/lyan/Documents/kaggle/severstal/crop_folds.csv',
                   fold=0):
    folds = pd.read_csv(folds_path)
    valid_ids_list = folds[folds.fold == fold].idx.values.tolist()
    train_ids_list = folds[folds.fold != fold].idx.values.tolist()

    trn_ds = SegData(train_ids_list, image_dir=image_dir, mask_dir=mask_dir, aug=aug_trn)
    val_ds = SegData(valid_ids_list, image_dir=image_dir, mask_dir=mask_dir, aug=aug_val)

    return trn_ds, val_ds


def cls_from_folds(image_dir, aug_trn, aug_val,
                   folds_path='/home/lyan/Documents/kaggle/severstal/crop_folds.csv',
                   fold=0):

    folds = pd.read_csv(folds_path)
    valid_ids_list = folds[folds.fold == fold]
    train_ids_list = folds[folds.fold != fold]

    trn_ds = SevClass(train_ids_list, image_dir=image_dir, aug=aug_trn)
    val_ds = SevClass(valid_ids_list, image_dir=image_dir, aug=aug_val)

    return trn_ds, val_ds


def create_model(model_name):
    if model_name == 'efficientnet-b0':
        return EfficientNet.from_pretrained()


if __name__ == '__main__':
    ds, _ = seg_from_folds('/var/ssd_1t/severstal/img_crops/', '/var/ssd_1t/severstal/mask_crops/', None, None)
    print(next(iter(ds)))

    ds, _ = seg_from_folds('/var/ssd_1t/severstal/img_crops/', None, None)
    print(next(iter(ds)))
