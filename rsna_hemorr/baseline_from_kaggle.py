from rsna_hemorr.hem_augs import transform_train, transform_test
from rsna_hemorr.hem_data import IntracranialDataset

dir_csv = '../input/rsna-intracranial-hemorrhage-detection'
dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'
dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'

n_classes = 6
n_epochs = 5
batch_size = 128

# Libraries

import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import Compose, VerticalFlip, HorizontalFlip
from albumentations.pytorch import ToTensor
from apex import amp
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm


train = pd.read_csv(os.path.join(dir_csv, 'stage_1_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_1_sample_submission.csv'))

train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
print(train.head())

png = glob.glob(os.path.join(dir_train_img, '*.png'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)

train = train[train['Image'].isin(png)]
train.to_csv('train.csv', index=False)

# Also prepare the test data

test[['ID','Image','Diagnosis']] = test['ID'].str.split('_', expand=True)
test['Image'] = 'ID_' + test['Image']
test = test[['Image', 'Label']]
test.drop_duplicates(inplace=True)

test.to_csv('test.csv', index=False)

# Data loaders

train_dataset = IntracranialDataset(
    csv_file='train.csv', path=dir_train_img, transform=transform_train, labels=True)

test_dataset = IntracranialDataset(
    csv_file='test.csv', path=dir_test_img, transform=transform_test, labels=False)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

transform_test_hf = Compose([
    HorizontalFlip(p=1, always_apply=True),
    ToTensor()
])

test_dataset_hf = IntracranialDataset(
    csv_file='test.csv', path=dir_test_img, transform=transform_test_hf, labels=False)

data_loader_test_hf = torch.utils.data.DataLoader(test_dataset_hf, batch_size=batch_size, shuffle=False, num_workers=4)

transform_test_vf = Compose([
    VerticalFlip(p=1, always_apply=True),
    ToTensor()
])

test_dataset_vf = IntracranialDataset(
    csv_file='test.csv', path=dir_test_img, transform=transform_test_vf, labels=False)

data_loader_test_vf = torch.utils.data.DataLoader(test_dataset_vf, batch_size=batch_size, shuffle=False, num_workers=4)

# Model

device = torch.device("cuda:0")
model = EfficientNet.from_pretrained('efficientnet-b2')
model._fc = torch.nn.Linear(1408, n_classes)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 2e-5}]
optimizer = optim.Adam(plist, lr=2e-5)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()
    tr_loss = 0

    tk0 = tqdm(data_loader_train, desc="Iteration")

    for step, batch in enumerate(tk0):
        inputs = batch["image"]
        labels = batch["labels"]

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        tr_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))


# Inference