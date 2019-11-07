import torchvision.transforms as transforms

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
train_aug = transforms.Compose([
   transforms.RandomResizedCrop(224),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   normalize
])

valid_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
