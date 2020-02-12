from typing import Callable

import numpy as np
from catalyst.core import Callback, CallbackOrder, _State
from sklearn.metrics import recall_score


class MyCustomMetric(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """

    def __init__(
            self,
            prefix='weight_recall',
            metric_fn=None,
            input_key: str = "targets",
            output_key: str = "logits",
            **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: _State):
        h1_score = calc_metric(state.output['h1_logits'], state.input['h1_targets'])
        h2_score = calc_metric(state.output['h2_logits'], state.input['h2_targets'])
        h3_score = calc_metric(state.output['h3_logits'], state.input['h3_targets'])

        metric = h1_score * 0.5 + h2_score * 0.25 + h3_score * 0.25

        state.metric_manager.add_batch_value(name=self.prefix, value=metric)


def calc_metric(pred, gt):
    pred = pred.detach().cpu().numpy()
    pred = np.argmax(pred, axis=1).astype(np.uint8)
    gt = gt.detach().cpu().numpy().astype(np.uint8)
    return recall_score(y_true=gt, y_pred=pred, average='macro')


score_callback = MyCustomMetric()