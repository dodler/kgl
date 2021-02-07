import torchvision.transforms as transforms
from PIL import Image


class PretrainDs:
    def __init__(self, df):
        self.df = df
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ,
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img_path = self.df.images.values[item]
        label = self.df.labels.values[item]

        img = Image.open(img_path).convert('RGB')
        img = self.aug(img)

        return img, label
