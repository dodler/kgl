import logging
from typing import List  # isort:skip

from catalyst.dl import CriterionCallback, State

from kaggle_lyan_utils import mixup, mixup_criterion
import torch.nn as nn

logger = logging.getLogger(__name__)

criterion = nn.CrossEntropyLoss(reduction='mean')


class MixupCallback(CriterionCallback):
    """
    Callback to do mixup augmentation.
    Paper: https://arxiv.org/abs/1710.09412
    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
            self,
            crit_key,
            input_key,
            output_key,
            fields: List[str] = ("features",),
            alpha=1.0,
            on_train_only=True,
            **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.crit_key = crit_key
        self.on_train_only = on_train_only
        self.fields = fields
        self.input_key = input_key
        self.output_key = output_key
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def _compute_loss(self, state: State, criterion):
        pred1 = state.output['h1_logits']
        pred2 = state.output['h2_logits']
        pred3 = state.output['h3_logits']

        if not self.is_needed:
            inp1 = state.input['h1_targets']
            inp2 = state.input['h2_targets']
            inp3 = state.input['h3_targets']
            return 0.7 * criterion(pred1, inp1) + 0.1 * criterion(pred2, inp2) + 0.2 * criterion(pred3, inp3)
        else:
            targets = state.input['targets']
            return mixup_criterion(pred1, pred2, pred3, targets)

    def on_loader_start(self, state: State):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")
        print('!!!!!!!!!!!!! on loader start', state.loader_name, self.is_needed)

    def on_batch_start(self, state: State):
        data = state.input['features']
        inp1 = state.input['h1_targets']
        inp2 = state.input['h2_targets']
        inp3 = state.input['h3_targets']

        if not self.is_needed:
            return

        data, targets = mixup(data, inp1, inp2, inp3, self.alpha)

        state.input['features'] = data
        state.input['targets'] = targets
