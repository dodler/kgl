from torch import Tensor
import torch
import numpy as np


def pytorch_neg_multi_log_likelihood_batch(
        gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    # assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    # assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    # assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    # assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # # assert all data are valid
    # assert torch.isfinite(pred).all(), "invalid value found in pred"
    # assert torch.isfinite(gt).all(), "invalid value found in gt"
    # assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    # assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
        gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


if __name__ == '__main__':
    import os
    from pathlib import Path

    import numpy as np
    import torch
    from l5kit.data import ChunkedDataset, LocalDataManager
    from l5kit.dataset import AgentDataset
    from l5kit.rasterization import build_rasterizer
    from torch.utils.data import DataLoader, RandomSampler

    from lyft_motion_prediction.models import LyftMultiModel, LyftMultiRegressor

    from tqdm import tqdm

    DIR_INPUT = "/var/ssd_2t_1/kaggle_lyft/"
    # DIR_INPUT = "/var/data/ssd/kaggle_lyft/"
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
            'raster_size': [512, 512],
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
            'batch_size': 48,
            'num_workers': 4
        },

        'valid_data_loader': {
            'key': 'scenes/validate.zarr',
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4
        },

        'train_params': {
            'max_num_steps': 364000 // 2,
            # 'max_num_steps': 100,
            'checkpoint_every_n_steps': 5000,

            # 'eval_every_n_steps': -1
        },
    }

    dm = LocalDataManager(None)

    train_cfg = cfg["train_data_loader"]

    rasterizer = build_rasterizer(cfg, dm)
    print('raster built')

    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    print('zarr opened')
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    print('dataset created')
    # train_dataloader = DataLoader(train_dataset,
    #                               sampler=RandomSampler(
    #                                   train_dataset,
    #                                   num_samples=cfg["train_params"]["max_num_steps"],
    #                                   replacement=True,
    #                               ),
    #                               batch_size=train_cfg["batch_size"],
    #                               num_workers=train_cfg["num_workers"])
    # print('datalaoder created')
    import time

    st = time.time()
    n = 100
    it = iter(train_dataset)
    for i in range(n):
        next(it)
    print('time spent', time.time() - st)
