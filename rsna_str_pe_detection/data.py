import glob

import cv2
from torch.utils.data.dataset import Dataset
from glob import glob

from rsna_str_pe_detection.augmentations import get_rsna_train_aug


class RsnaDS(Dataset):
    def __init__(self, df, aug, path):
        self.aug = aug
        self.df = df
        self.path = path
        self.fnames = self.df[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]
        self.labels = self.df[['pe_present_on_image', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1',
                               'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe', 'rightsided_pe',
                               'acute_and_chronic_pe', 'central_pe', 'indeterminate']]

    def __getitem__(self, idx):
        stuid = self.fnames.loc[idx].values[0]
        siuid = self.fnames.loc[idx].values[1]
        souid = self.fnames.loc[idx].values[2]
        img_path = glob.glob('{}/train-jpegs/{}/{}/*{}.jpg'.format(self.path, stuid, siuid, souid))[0]
        img = cv2.imread(img_path)

        y = self.labels.loc[idx].values
        img = self.aug(image=img)['image']
        return img, y

    def __len__(self):
        return self.df.shape[0]


if __name__ == '__main__':
    import pandas as pd

    DIR_INPUT = '/var/ssd_2t_1/kaggle_rsna_str_pe/'

    aug = get_rsna_train_aug('v0')

    data = pd.read_csv('{}train_random_fold.csv'.format(DIR_INPUT))
    fold_data = data[data.fold != 0]
    fold_data = fold_data.reset_index().drop('index', axis=1)
    train_dataset = RsnaDS(
        df=fold_data,
        aug=aug,
        path=DIR_INPUT,
    )
    it = iter(train_dataset)
    next(it)
    raise Exception()

    import time

    n = 100
    st = time.time()

    for i in tqdm(range(n)):
        next(it)
    print('dataset time spent', time.time() - st)

    import torch
    from torch.utils.data import RandomSampler

    sampler = RandomSampler(
        train_dataset,
        num_samples=100000,
        replacement=True,
    )

    loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=128, num_workers=8)

    it = iter(loader)

    import time

    st = time.time()

    for i in tqdm(range(n)):
        next(it)
    print('dataloader time spent', time.time() - st)
