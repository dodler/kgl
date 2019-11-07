import pickle
import numpy as np

import os
import os.path as osp
from tqdm import *

import pandas as pd

from clouds_sat.cloud_utils import mask2rle, post_process, sigmoid

all_files = os.listdir('/var/data/clouds_valid_masks/')

valid_prob_files = [k for k in all_files if 'valid_prob' in k]
valid_prob_files = [osp.join('/var/data/clouds_valid_masks/', k) for k in valid_prob_files]

test_prob_files = [k for k in all_files if 'test_prob' in k]
test_prob_files = [osp.join('/var/data/clouds_valid_masks/', k) for k in test_prob_files]

class_param_files = [k for k in all_files if 'class_param' in k]
class_param_files = [osp.join('/var/data/clouds_valid_masks/', k) for k in class_param_files]


def load_and_sum(file_list):
    print('loading from list:', file_list)
    probs = np.load(file_list[0])
    for vp_file in tqdm(file_list[1:]):
        probs += np.load(vp_file)
    return probs


test_prob_sum = load_and_sum(test_prob_files) / len(test_prob_files)

class_param_list = []
for file in class_param_files:
    with open(file, 'rb') as f:
        class_param_list.append(pickle.load(f))

mean_class_param = {}
class_means = [0] * 4
for cls_param in class_param_list:
    for cls in cls_param:
        class_means[cls] += cls_param[cls][0]

class_means = [k / len(class_param_list) for k in class_means]

for cls in cls_param:
    mean_class_param[cls] = (class_means[cls], cls_param[cls][1])

path = '/var/ssd_1t/cloud/'
train = pd.read_csv(f'{path}/train.csv')
sub = pd.read_csv(f'{path}/sample_submission.csv')

n_train = len(os.listdir(f'{path}/train_images'))
n_test = len(os.listdir(f'{path}/test_images'))
print(f'There are {n_train} images in train dataset')
print(f'There are {n_test} images in test dataset')
sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

class_params = mean_class_param

encoded_pixels = []
image_id = 0

for probability in tqdm(test_prob_sum):
    predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                        class_params[image_id % 4][1])
    if num_predict == 0:
        encoded_pixels.append('')
    else:
        r = mask2rle(predict)
        encoded_pixels.append(r)
    image_id += 1

sub['EncodedPixels'] = encoded_pixels
sub.to_csv('submission.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
