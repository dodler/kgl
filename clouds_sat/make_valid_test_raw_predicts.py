import argparse
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import tqdm
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner
from torch.utils.data import DataLoader

from clouds_sat.cloud_data import CloudDataset
from clouds_sat.cloud_utils import get_preprocessing, get_validation_augmentation, post_process, sigmoid, dice
from segmentation.custom_unet import Unet

path = '/var/ssd_1t/cloud/'
train = pd.read_csv(f'{path}/train.csv')
sub = pd.read_csv(f'{path}/sample_submission.csv')

n_train = len(os.listdir(f'{path}/train_images'))
n_test = len(os.listdir(f'{path}/test_images'))
print(f'There are {n_train} images in train dataset')
print(f'There are {n_test} images in test dataset')

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

ACTIVATION = 'sigmoid'
model = Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=4,
    activation=ACTIVATION,
)

folds_path = 'stage_1_train_folds.csv'
fold = 0
folds = pd.read_csv('/home/lyan/Documents/kaggle/clouds_sat/train_folds.csv')

main_col = 'Image_Label'
train = pd.read_csv(f'{path}/train.csv')
train['label'] = train[main_col].apply(lambda x: x.split('_')[1])
train['im_id'] = train[main_col].apply(lambda x: x.split('_')[0])

valid_ids = folds[folds.fold == fold]['img_id'].values

valid_dataset = CloudDataset(path=path, df=train,
                             datatype='valid',
                             img_ids=valid_ids,
                             transforms=get_validation_augmentation(),
                             preprocessing=get_preprocessing(preprocessing_fn))

parser = argparse.ArgumentParser('Make validation masks')
parser.add_argument('--num-workers', type=int, default=4, required=False)
parser.add_argument('--batch-size', type=int, default=4, required=False)
parser.add_argument('--logdir', type=str, required=True)
args = parser.parse_args()

valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

runner = SupervisedRunner()

logdir = args.logdir

encoded_pixels = []
loaders = {"infer": valid_loader}
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ],
)

valid_masks = []
valid_probabilities = np.zeros((len(valid_dataset) * 4, 350, 525))
for i, (batch, output) in enumerate(tqdm.tqdm(zip(
        valid_dataset, runner.callbacks[0].predictions["logits"]))):
    image, mask = batch
    for m in mask:
        if m.shape != (350, 525):
            m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        valid_masks.append(m)

    for j, probability in enumerate(output):
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        valid_probabilities[i * 4 + j, :, :] = probability

class_params = {}
for class_id in range(4):
    print(class_id)
    attempts = []
    for t in range(0, 100, 5):
        t /= 100
        for ms in [0, 100, 1200, 5000, 10000]:
            masks = []
            for i in range(class_id, len(valid_probabilities), 4):
                probability = valid_probabilities[i]
                predict, num_predict = post_process(sigmoid(probability), t, ms)
                masks.append(predict)

            d = []
            for i, j in zip(masks, valid_masks[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))

            attempts.append((t, ms, np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    class_params[class_id] = (best_threshold, best_size)

valid_masks = np.array(valid_masks)

test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

test_dataset = CloudDataset(path=path, df=sub, datatype='test', img_ids=test_ids,
                            transforms=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=10)

print('doing test infer')
loaders = {"test": test_loader}
encoded_pixels = []
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ],
)

encoded_pixels = []
image_id = 0
test_predicts = []
for i, test_batch in tqdm.tqdm(enumerate(runner.callbacks[0].predictions['logits'])):
    runner_out = test_batch
    for i, probability in enumerate(runner_out):
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        test_predicts.append(probability)

test_predicts = np.array(test_predicts)

experiment_name = args.logdir.split('/')[-1]

np.save('/var/data/clouds_valid_masks/valid_masks_' + experiment_name + '.npy', valid_masks)
np.save('/var/data/clouds_valid_masks/valid_probabilities_' + experiment_name + '.npy', valid_probabilities)
np.save('/var/data/clouds_valid_masks/test_probabilities_' + experiment_name + '.npy', test_predicts)

with open('/var/data/clouds_valid_masks/class_params_'+experiment_name+'.pkl','wb') as f:
    pickle.dump(class_params, f)
