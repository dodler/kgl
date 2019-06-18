import cv2
from albumentations import VerticalFlip, RandomCrop, PadIfNeeded, HorizontalFlip, Resize
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from engine import train_one_epoch, evaluate
import utils

from google_object_detection.data import get_aug, GoogleObjDetection

model=fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 600  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# model=torch.nn.DataParallel(model)
model.to(0)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

device=0
base_dir = '/var/ssd_1t/open_images_obj_detection/'
aug = get_aug([HorizontalFlip(p=0.5),
               Resize(768,768)])
dataset = GoogleObjDetection(base_dir, 'train', aug)
dataset_test = GoogleObjDetection(base_dir, 'train_small', aug)

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=6, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    torch.save(model.state_dict(), f'faster_rcnn_resnet50_epoch_{i}.pkl')
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

