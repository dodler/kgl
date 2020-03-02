from collections import OrderedDict
import torch
from catalyst.dl import SupervisedRunner, AccuracyCallback, EarlyStoppingCallback, MultiMetricCallback
from efficientnet_pytorch import EfficientNet
import pandas as pd

from common.metrics import catalyst_roc_auc, catalyst_logloss, catalyst_acc_score
from common.sampler import BalancedBatchSampler
from deep_fake.augs.v0 import train_aug, valid_aug
from deep_fake.deep_fake_data import DeepFakeDs

FOLD = 0

full_df = pd.read_csv('/home/lyan/Documents/kaggle/deep_fake/fakes_folds.csv', index_col=0)
valid_df = full_df[full_df.fold == FOLD]
train_df = full_df[full_df.fold != FOLD]
train_ds = DeepFakeDs(df=train_df, aug=train_aug)
valid_ds = DeepFakeDs(df=valid_df, aug=valid_aug)

sampler = BalancedBatchSampler(dataset=train_ds, labels=train_ds.labels)

train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=32, num_workers=10)
valid_loader = torch.utils.data.DataLoader(valid_ds, shuffle=False, batch_size=32, num_workers=10)


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x)


model = EfficientNet.from_pretrained('efficientnet-b3')
model._fc = torch.nn.Sequential(
    torch.nn.Linear(1536, 1),
    Squeeze(),
)

experiment_name = 'deepfake_crops_effnet_b3_224_v1'

lr = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, mode='min')
criterion = torch.nn.BCEWithLogitsLoss()
loaders = OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = valid_loader

num_epochs = 50
logdir = "/var/data/deepfake/" + experiment_name
runner = SupervisedRunner()

runner.train(
    fp16=False,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    logdir=logdir,
    scheduler=scheduler,
    num_epochs=num_epochs,
    callbacks=[
        MultiMetricCallback(metric_fn=catalyst_roc_auc, prefix='rocauc',
                            input_key="targets",
                            output_key="logits",
                            list_args=['_']),
        MultiMetricCallback(metric_fn=catalyst_logloss, prefix='logloss',
                            input_key="targets",
                            output_key="logits",
                            list_args=['_']),
        MultiMetricCallback(metric_fn=catalyst_acc_score, prefix='acc',
                            input_key="targets",
                            output_key="logits",
                            list_args=['_']),
        EarlyStoppingCallback(patience=10, min_delta=0.01)
    ],
    verbose=True
)
