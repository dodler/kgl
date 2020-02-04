import torch
from catalyst.core import MultiMetricCallback
from sklearn.metrics import recall_score
import numpy as np


def calc_metric(pred, gt, *args, **kwargs):
    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=1).astype(np.uint8)
    gt = gt.detach().cpu().numpy().astype(np.uint8)
    return [recall_score(y_true=gt, y_pred=pred, average='macro')]


h1_recall = MultiMetricCallback(metric_fn=calc_metric, prefix='h1_ma_rec',
                                input_key="h1_targets",
                                output_key="h1_logits",
                                list_args=['_'])

h2_recall = MultiMetricCallback(metric_fn=calc_metric, prefix='h2_ma_rec',
                                input_key="h2_targets",
                                output_key="h2_logits",
                                list_args=['_'])

h3_recall = MultiMetricCallback(metric_fn=calc_metric, prefix='h3_ma_rec',
                                input_key="h3_targets",
                                output_key="h3_logits",
                                list_args=['_'])
