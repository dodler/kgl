from monai.transforms import LoadNifti, Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, RandAffine
from monai.utils import get_seed
import torch
import numpy as np
import os
import cv2
import glob

from rsna_str_pe_detection.aug_3d import get_rsna_train_aug

target_cols = [
    'negative_exam_for_pe',  # exam level
    'rv_lv_ratio_gte_1',  # exam level
    'rv_lv_ratio_lt_1',  # exam level
    'leftsided_pe',  # exam level
    'chronic_pe',  # exam level
    'rightsided_pe',  # exam level
    'acute_and_chronic_pe',  # exam level
    'central_pe',  # exam level
    'indeterminate'  # exam level
]


class RSNADataset3D(torch.utils.data.Dataset, Randomizable):
    def __init__(self, df, path, transform=None):

        self.path = path
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def randomize(self) -> None:
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index):
        self.randomize()
        row = self.df.iloc[index]
        img_path = os.path.join(self.path, row.StudyInstanceUID, row.SeriesInstanceUID, '*.jpg')
        jpg_lst = sorted(glob.glob(img_path))
        img_lst = [cv2.imread(jpg)[:, :, ::-1] for jpg in jpg_lst]
        img = np.stack([image.astype(np.float32) for image in img_lst], axis=2).transpose(3, 0, 1, 2)

        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            img = apply_transform(self.transform, img)

        return img, torch.tensor(row[target_cols]).float()


if __name__ == '__main__':
    import pandas as pd

    DIR_INPUT = '/var/ssd_2t_1/kaggle_rsna_str_pe/'

    aug = get_rsna_train_aug('v0')

    data = pd.read_csv('{}train_debug.csv'.format(DIR_INPUT))
    print('data read done')
    fold_data = data[data.fold != 0]
    fold_data = fold_data.reset_index().drop('index', axis=1)
    train_dataset = RSNADataset3D(
        df=fold_data,
        transform=aug,
        path='{}/train-jpegs/'.format(DIR_INPUT),
    )
    it = iter(train_dataset)
    x, y = next(it)
    print(x.shape, y.shape)
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
