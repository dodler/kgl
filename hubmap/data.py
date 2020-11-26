import cv2
import numpy as np
import torch
import glob


class SegHubmapDs:
    def __init__(self, aug, path, train_ids):
        self.aug = aug
        self.train_ids = frozenset(train_ids)
        images = glob.glob('{}*'.format(path))
        print('len images', len(images))
        image_names = [k.split('/')[-1] for k in images]
        image_names = [k for k in image_names if k.split('_')[0] in self.train_ids]
        image_names = ['{}/{}'.format(path, k) for k in image_names]
        self.paths = image_names

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        mask_path = img_path.replace('images', 'masks')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        _aug = self.aug(image=img, mask=mask)
        img = _aug['image']
        mask = _aug['mask']

        return img, mask.float()


if __name__ == '__main__':
    import albumentations as alb
    from albumentations.pytorch.transforms import ToTensorV2

    aug = alb.Compose([
        alb.Resize(512, 512, p=1),
        alb.Normalize(p=1),
        ToTensorV2(p=1),
    ])
    ds = SegHubmapDs(aug=aug, path='input/crops/images/', train_ids=['2f6ecfcdf', 'aaa6a05cc', 'cb2d976f4', 'e79de561c','095bf7a1f', '54f2eec69', '1e2425f28'])
    print(len(ds))
    img, mask = next(iter(ds))
    print(img.shape, mask.shape)
