import torch
import random

from deep_fake.deep_fake_data import DeepFakeDs
import pandas as pd


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, idx):
        return self.labels[idx]

    def __len__(self):
        return self.balanced_max * len(self.keys)


if __name__ == '__main__':
    full_df = pd.read_csv('/home/lyan/Documents/kaggle/deep_fake/train_crops.csv', index_col=0)
    valid_df = full_df[full_df.fold == 0]
    train_df = full_df[full_df.fold != 1]
    train_ds = DeepFakeDs(df=train_df)
    valid_ds = DeepFakeDs(df=valid_df)

    sampler = BalancedBatchSampler(dataset=train_ds, labels=train_ds.labels)
    si = iter(sampler)
    batch = []
    for i in range(20):
        item = next(si)
        batch.append(train_ds[item][1])

    print(batch)
