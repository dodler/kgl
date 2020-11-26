from argparse import ArgumentParser

import torch.nn as nn
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader, RandomSampler
from warmup_scheduler import GradualWarmupScheduler
from adamp import AdamP

from lyft_motion_prediction.models import LyftMultiModel, LyftMultiRegressor

parser = ArgumentParser()
parser.add_argument('--resume', type=str, required=False, default=None)
parser.add_argument('--cfg', type=str, required=True, default=None)

DIR_INPUT = "./input/"
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

device = torch.device('cuda')

DEBUG = False

cfg = None
print(cfg)

dm = LocalDataManager(None)


class LyftModule(pl.LightningModule):

    def __init__(self, cfg_path):
        super().__init__()

        self.opt = None
        self.sched = None

        self.hparams.warm_up_step = 2
        self.hparams.cfg_path = cfg_path

        model = LyftMultiModel(cfg)
        self.predictor = LyftMultiRegressor(model)

        self.pred_coords_list = []
        self.confidences_list = []
        self.timestamps_list = []
        self.track_id_list = []

    def forward(self, x):
        return self.predictor(x)

    def get_lr(self):
        """ Returns current learning rate for schedulers """

        if self.opt is None:
            raise ValueError('No learning rate schedulers initialized')
        else:
            for pg in self.opt.param_groups:
                return pg['lr']

    def training_step(self, data, batch_nb):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)

        loss, _ = self.predictor(inputs, targets, target_availabilities)

        tensorboard_logs = {
            'train/_loss': loss,
            'train/_lr': self.get_lr(),
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, data, batch_nb):
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].to(device)
        targets = data["target_positions"].to(device)
        world_from_agents = data["world_from_agent"].cpu().numpy()
        centroid = data["centroid"].cpu().numpy()

        pred, confidences = self.predictor.predictor(inputs)

        pred = pred.cpu().numpy()
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                pred[i, j, :] = transform_points(pred[i, j,], world_from_agents[i]) - centroid[i, :2]

        self.pred_coords_list.append(pred)
        self.confidences_list.append(confidences.cpu().numpy().copy())
        self.timestamps_list.append(data["timestamp"].cpu().numpy().copy())
        self.track_id_list.append(data["track_id"].cpu().numpy().copy())

        return {'val_loss': 0}

    def validation_epoch_end(self, outputs):
        timestamps = np.concatenate(self.timestamps_list)
        track_ids = np.concatenate(self.track_id_list)
        coords = np.concatenate(self.pred_coords_list)
        confs = np.concatenate(self.confidences_list)

        self.timestamps_list = []
        self.track_id_list = []
        self.pred_coords_list = []
        self.confidences_list = []

        csv_path = "val.csv"
        write_pred_csv(
            csv_path,
            timestamps=timestamps,
            track_ids=track_ids,
            coords=coords,
            confs=confs)

        try:
            metrics = compute_metrics_csv(self.eval_gt_path, csv_path, [neg_multi_log_likelihood])
            target_metric = 0
            for metric_name, metric_mean in metrics.items():
                target_metric = metric_mean
                break
        except:
            target_metric = 1000

        print('got target metric', target_metric)
        tensorboard_logs = {'val/_metric': target_metric}

        return {'avg_val_loss': target_metric, 'log': tensorboard_logs}

    def configure_optimizers(self):
        lr = float(cfg["train_params"]["lr"])
        if cfg['optimizer'] == 'adam':
            opt = torch.optim.Adam(self.predictor.parameters(), lr=lr, weight_decay=5e-4)
        if cfg['optimizer'] == 'adamw':
            opt = torch.optim.AdamW(self.predictor.parameters(), lr=lr, weight_decay=5e-4)
        elif cfg['optimizer'] == 'adamp':
            opt = AdamP(self.predictor.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=1e-2)

        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.98 ** epoch
            lr_scale = max(1e-6, lr_scale)

            return lr_scale

        scheduler = LambdaLR(
            opt,
            lr_lambda=lr_foo
        )

        self.opt = opt
        return opt  # , [scheduler]
        # return [opt], [scheduler]

    def train_dataloader(self):
        train_cfg = cfg["train_data_loader"]

        rasterizer = build_rasterizer(cfg, dm)

        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(
                                          train_dataset,
                                          num_samples=cfg["train_params"]["max_num_steps"],
                                          replacement=True,
                                      ),
                                      batch_size=train_cfg["batch_size"],
                                      num_workers=train_cfg["num_workers"])
        return train_dataloader

    def val_dataloader(self):
        # created chopped dataset

        rasterizer = build_rasterizer(cfg, dm)
        eval_cfg = cfg["valid_data_loader"]
        num_frames_to_chop = 100
        eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]),
                                                cfg["raster_params"]["filter_agents_threshold"],
                                                num_frames_to_chop, cfg["model_params"]["future_num_frames"],
                                                MIN_FUTURE_STEPS)

        eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
        eval_mask_path = str(Path(eval_base_path) / "mask.npz")
        eval_gt_path = str(Path(eval_base_path) / "gt.csv")
        self.eval_gt_path = eval_gt_path

        eval_zarr = ChunkedDataset(eval_zarr_path).open(cache_size_bytes=10e9)
        eval_mask = np.load(eval_mask_path)["arr_0"]

        eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
        eval_dataloader = DataLoader(eval_dataset,
                                     shuffle=False,
                                     batch_size=eval_cfg["batch_size"],
                                     num_workers=8)

        return eval_dataloader


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config_data(args.cfg)
    cfg_name = args.cfg.split('/')[-1]

    if args.cfg is None:
        raise Exception('Cfg not specified')

    module = LyftModule(args.cfg)
    if args.resume is not None:
        print("loading from", args.resume)
        model = LyftModule.load_from_checkpoint(args.resume)
    default_root_dir = '/var/data/hdd1/{}/lyft_checkpoints/'.format(cfg_name)

    checkpoint_callback = ModelCheckpoint(
        filepath=default_root_dir,
        # filepath='/var/data/lyft_checkpoints/',
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix='_'
    )

    early_stop = EarlyStopping(monitor='avg_val_loss', verbose=True, patience=10, mode='min')
    print('using default root dir', default_root_dir)
    trainer = pl.Trainer(gpus=1, max_epochs=cfg['train_params']['epochs'],
                         default_root_dir=default_root_dir,
                         callbacks=[early_stop,checkpoint_callback ])
    trainer.fit(module)
