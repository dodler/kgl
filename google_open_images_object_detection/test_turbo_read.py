import cv2
from tqdm import *
import os
import os.path as osp

from kaggle_lyan_utils import read_turbo

if __name__=="__main__":
    base_dir = '/var/ssd_1t/open_images_obj_detection/'
    for im in tqdm(os.listdir(base_dir+'train')):
        img_fp=osp.join(base_dir, 'train',im)
        try:
            img=read_turbo(img_fp)
        except:
            print(img_fp)
        # img=cv2.imread(img_fp)
