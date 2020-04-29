import cv2
import torch


class DeepFakeDs:
    def __init__(self, df, aug=None):
        assert df is not None
        assert df.columns[0] == 'label'

        self.aug = aug
        self.df = df
        self.labels = self.df.label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img_path = self.df.index[item]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.df.label[10]

        if self.aug is not None:
            img = self.aug(image=img)['image']

        return img, torch.tensor(label).float()


class ImageListDs:
    def __init__(self, images, labels, aug):
        self.images = images
        self.aug = aug
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.images[idx]
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)['image']

        return label, img
