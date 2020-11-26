import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

DIR_INPUT = "/var/ssd_2t_1/kaggle_lyft/"

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

DEBUG = False

cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'effb5',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [300, 300],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train_full.zarr',
        # 'key': 'scenes/train.zarr',
        'batch_size': 24,
        'shuffle': True,
        'num_workers': 6
    },

    'train_params': {
        'max_num_steps': 100 if DEBUG else 50000,
        'checkpoint_every_n_steps': 5000,

        # 'eval_every_n_steps': -1
    }
}

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset,
                              shuffle=train_cfg["shuffle"],
                              batch_size=train_cfg["batch_size"],
                              num_workers=train_cfg["num_workers"])

print(train_dataset)


class LyftModel(nn.Module):

    def __init__(self, cfg: Dict):
        super().__init__()

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone = EfficientNet.from_pretrained('efficientnet-b4', in_channels=num_in_channels)

        backbone_out_features = 1792

        num_targets = 2 * cfg["model_params"]["future_num_frames"]

        # You can add more layers here.
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.logit = nn.Linear(4096, out_features=num_targets)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        b, c, d, d = x.shape
        x = F.adaptive_avg_pool2d(x, 1).squeeze()

        x = self.head(x)
        x = self.logit(x)

        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_output_name = 'eff_b4_full'

if __name__ == '__main__':

    model = LyftModel(cfg)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss(reduction="none")

    tr_it = iter(train_dataloader)

    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    losses_train = []

    for itr in progress_bar:

        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)

        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)

        outputs = model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)

        loss = loss * target_availabilities
        loss = loss.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())

        if (itr + 1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:
            torch.save(model.state_dict(), '{}_iter_{}.pth'.format(model_output_name, itr))

        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train[-100:])}")