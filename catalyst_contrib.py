import logging
from typing import List  # isort:skip

import numpy as np
import torch
from catalyst.dl import CriterionCallback, State

from kaggle_lyan_utils import mixup

logger = logging.getLogger(__name__)


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
        pred2 = state.output['h1_logit2']
        pred3 = state.output['h1_logit3']
        data = state.input['features']

        return mixup(data, pred1, pred2, pred3)

    def on_loader_start(self, state: State):
        self.is_needed = not self.on_train_only or \
                         state.loader_name.startswith("train")

    def on_batch_start(self, state: State):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                             (1 - self.lam) * state.input[f][self.index]
