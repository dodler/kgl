import os
import numpy as np
import torch
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.evaluation import write_pred_csv
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from lyft_motion_prediction.models import LyftMultiModel
import copy

from lyft_motion_prediction.multi_mode_train import LyftModule

torch.set_grad_enabled(False)
torch.set_num_threads(2)

DIR_INPUT = "./input"
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"
MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, required=True, default=None)
parser.add_argument('--cfg', type=str, required=True, default=None)
args = parser.parse_args()

cfg = load_config_data(args.cfg)

device = 'cuda'
if not torch.cuda.is_available():
    print('cuda not available, using cpu')
    device = 'cpu'


def run_prediction(predictor, data_loader):
    predictor.eval()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            image = data["image"].to(device)
            pred, confidences = predictor(image)

            world_from_agents = data["world_from_agent"].cpu().numpy()
            centroid = data["centroid"].cpu().numpy()

            pred = pred.cpu().numpy()
            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    pred[i, j, :] = transform_points(pred[i, j,], world_from_agents[i]) - centroid[i, :2]

            pred_coords_list.append(pred)
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    return timestamps, track_ids, coords, confs


flags_dict = {
    "debug": False,
    "l5kit_data_folder": DIR_INPUT,
    "pred_mode": "multi",
    "device": "cuda:0",
    "out_dir": "results/multi_train",
    "epoch": 2,
    "snapshot_freq": 50,
}

print("Load dataset...")
default_test_cfg = {
    'key': 'scenes/test.zarr',
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 12
}
test_cfg = cfg.get("test_data_loader", default_test_cfg)

rasterizer = build_rasterizer(cfg, dm)

test_path = test_cfg["key"]
print(f"Loading from {test_path}")
test_zarr = ChunkedDataset(dm.require(test_path)).open()
print("test_zarr", type(test_zarr))
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataset = test_agent_dataset

test_loader = DataLoader(
    test_dataset,
    shuffle=test_cfg["shuffle"],
    batch_size=test_cfg["batch_size"],
    num_workers=test_cfg["num_workers"],
    pin_memory=True,
)

print(test_agent_dataset)
print("# AgentDataset test:", len(test_agent_dataset))
print("# ActualDataset test:", len(test_dataset))

args = parser.parse_args()
cfg = load_config_data(args.cfg)

module = LyftModule(args.cfg)
module.load_state_dict(torch.load(args.resume, map_location='cpu'))
predictor = module.predictor

timestamps, track_ids, coords, confs = run_prediction(predictor, test_loader)

csv_path = "submission.csv"
write_pred_csv(
    csv_path,
    timestamps=timestamps,
    track_ids=track_ids,
    coords=coords,
    confs=confs)
print(f"Saved to {csv_path}")
