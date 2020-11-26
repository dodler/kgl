import cv2


class CassavaDs:
    def __init__(self, df, aug, path):
        self.df = df
        self.aug = aug
        self.path = path

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_idx = self.df.iloc[idx, 0]
        image_path = '{}/{}'.format(self.path, image_idx)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)['image']
        label = self.df.iloc[idx, 1]

        return img, label
