import torch
import pandas as pd

from severstal.sev_data import SegData


def make_experiment_name(args, conf):
    comment = ''
    if conf.comment is not None:
        comment = conf.comment

    result = conf.backbone + '_' \
             + conf.seg_net + '_' \
             + str(conf.opt) + '_' \
             + str(args.batch_size) + '_' \
             + str(conf.loss) + '_' \
             + 'fold' + str(args.fold) + '_' \
             + comment + '_'

    if conf.opt_step_size > 1:
        result += '_step_size_' + str(args.opt_step_size) + '_'

    if conf.swa:
        result += '_swa_'

    if conf.enorm:
        result += '_enorm_'

    if conf.pseudo_label:
        result += '_pseudo_label_'

    if conf.backbone_weights is not None:
        result += '_backbone_weights_' + args.backbone_weights.replace('/', '_') + '_'

    result += datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    return result.replace('__', '_')


def load_pretrained_weights(path, model):
    pretrained_dict = torch.load(path, map_location='cpu')

    model_dict = model.encoder.state_dict()
    pretrained_dict = {k.replace('feature_extr.', ''): v for k, v in pretrained_dict.items()
                       if k.replace('feature_extr.', '') in model_dict}
    model_dict.update(pretrained_dict)
    if 'se_resne' in path:
        model_dict['last_linear.bias'] = None
        model_dict['last_linear.weight'] = None
    model.encoder.load_state_dict(model_dict)


def from_folds(image_dir,
               mask_dir,
               aug_trn,
               aug_val,
               folds_path='/home/lyan/Documents/kaggle/severstal/crop_folds.csv',
               fold=0):
    folds = pd.read_csv(folds_path)
    valid_ids_list = folds[folds.fold == fold].ids.values.tolist()
    train_ids_list = folds[folds.fold != fold].ids.values.tolist()

    valid_ids_list = [k.split('/')[-1] for k in valid_ids_list]
    train_ids_list = [k.split('/')[-1] for k in train_ids_list]

    trn_ds = SegData(train_ids_list, image_dir=image_dir, mask_dir=mask_dir, aug=aug_trn)
    val_ds = SegData(valid_ids_list, image_dir=image_dir, mask_dir=mask_dir, aug=aug_val)

    trn_ds.img_ids = train_ids_list
    val_ds.img_ids = valid_ids_list

    return trn_ds, val_ds
