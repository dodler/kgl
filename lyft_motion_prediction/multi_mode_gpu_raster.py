import os
from pathlib import Path

import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from adamp import AdamP
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.evaluation import write_pred_csv, compute_metrics_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood
from l5kit.rasterization import build_rasterizer
from l5kit.rasterization.rasterizer_builder import _load_metadata, get_hardcoded_world_to_ecef
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler

from lyft_motion_prediction.models import LyftMultiModel, LyftMultiRegressor

# DIR_INPUT = "/var/ssd_2t_1/kaggle_lyft/"
from lyft_motion_prediction.opengl_raster import OpenGLSemanticRasterizer

DIR_INPUT = "input/"
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

device = torch.device('cuda')

DEBUG = False

eval_base_path = '{}scenes/validate_chopped_100'.format(DIR_INPUT)
eval_gt_path = str(Path(eval_base_path) / "gt.csv")

cfg = {
    'optimizer': "adamp",
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [224, 224],
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
        'batch_size': 16,
        'num_workers': 8
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 8
    },

    'train_params': {
        'max_num_steps': 364000 // 2,
        # 'max_num_steps': 100,
        'checkpoint_every_n_steps': 5000,

        # 'eval_every_n_steps': -1
    },
}

dm = LocalDataManager(None)


class LyftModule(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.opt = None
        self.sched = None

        self.hparams.warm_up_step = 2

        model = LyftMultiModel(cfg)
        self.predictor = LyftMultiRegressor(model)

        self.pred_coords_list = []
        self.confidences_list = []
        self.timestamps_list = []
        self.track_id_list = []

    def forward(self, x):
        return self.model(x)

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

        pred, confidences = self.predictor.predictor(inputs)

        self.pred_coords_list.append(pred.cpu().numpy().copy())
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
            metrics = compute_metrics_csv(eval_gt_path, csv_path, [neg_multi_log_likelihood])
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
        if cfg['optimizer'] == 'adam':
            opt = torch.optim.Adam(self.predictor.parameters(), lr=5e-3, weight_decay=5e-4)
        elif cfg['optimizer'] == 'adamp':
            opt = AdamP(self.predictor.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-2)

        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = LambdaLR(
            opt,
            lr_lambda=lr_foo
        )

        self.sched = scheduler
        self.opt = opt
        return [opt], [scheduler]

    def train_dataloader(self):
        train_cfg = cfg["train_data_loader"]

        try:
            dataset_meta = _load_metadata(train_cfg["dataset_meta_key"], dm)
            world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
        except (KeyError, FileNotFoundError):
            world_to_ecef = get_hardcoded_world_to_ecef()

        semantic_map_filepath = dm.require(train_cfg["semantic_map_key"])

        rasterizer = OpenGLSemanticRasterizer(
            raster_size=train_cfg["raster_size"],
            pixel_size=train_cfg["pixel_size"],
            ego_center=train_cfg["ego_center"],
            filter_agents_threshold=0.5,
            history_num_frames=train_cfg['history_num_frames'],
            semantic_map_path=semantic_map_filepath,
            world_to_ecef=world_to_ecef,
        )

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
        eval_cfg = cfg["valid_data_loader"]
        rasterizer = build_rasterizer(cfg, dm)
        eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
        eval_mask_path = str(Path(eval_base_path) / "mask.npz")
        eval_gt_path = str(Path(eval_base_path) / "gt.csv")

        eval_zarr = ChunkedDataset(eval_zarr_path).open()
        eval_mask = np.load(eval_mask_path)["arr_0"]

        eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
        eval_dataloader = DataLoader(eval_dataset,
                                     shuffle=False,
                                     batch_size=eval_cfg["batch_size"],
                                     num_workers=8)

        return eval_dataloader


if __name__ == '__main__':
    module = LyftModule()

    checkpoint_callback = ModelCheckpoint(
        filepath='/var/data/hdd1/lyft_checkpoints/',
        # filepath='/var/data/lyft_checkpoints/',
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
        prefix='_'
    )

    early_stop = EarlyStopping(monitor='avg_val_loss', verbose=True, patience=10, mode='min')
    trainer = pl.Trainer(gpus=1, max_epochs=50,
                         default_root_dir='/var/data/hdd1/lyft_checkpoints/',
                         # default_root_dir='/var/data/lyft_checkpoints/',
                         early_stop_callback=early_stop,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(module)
