import cv2


class ShopeeDs:
    def __init__(self, path, images, labels, aug):
        self.path = path
        self.images = images
        self.labels = labels
        self.aug = aug

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = self.images[item]
        img_path = '{}/{}'.format(self.path, img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.aug(image=img)['image']

        label = self.labels[item]

        return img, label, item
