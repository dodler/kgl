import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY
from typing import Optional, Any, List, Union

import torch
from catalyst.dl.core import Callback, RunnerState, CallbackOrder

jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0')


def read_turbo(path):
    with open(path, 'rb') as f:
        img = jpeg.decode(f.read())
    return img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


class CriterionAggregatorCallback(Callback):
    """
    This callback allows you to aggregate the values of the loss
    (by ``sum`` or ``mean``) and put the value back into ``state.loss``.
    """

    def __init__(
            self,
            prefix: str,
            loss_keys: Union[str, List[str]] = None,
            loss_aggregate_fn: str = "sum",
            multiplier: float = 1.0
    ) -> None:
        """
        Args:
            prefix (str): new key for aggregated loss.
            loss_keys (List[str]): If not empty, it aggregates
                only the values from the loss by these keys.
            loss_aggregate_fn (str): function for aggregation.
                Must be either ``sum`` or ``mean``.
            multiplier (float): scale factor for the aggregated loss.
        """
        super().__init__(CallbackOrder.Criterion + 1)
        assert prefix is not None and isinstance(prefix, str), \
            "prefix must be str"
        self.prefix = prefix

        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        self.loss_keys = loss_keys

        self.multiplier = multiplier
        if loss_aggregate_fn == "sum":
            self.loss_fn = lambda x: torch.sum(torch.stack(x)) * multiplier
        elif loss_aggregate_fn == "mean":
            self.loss_fn = lambda x: torch.mean(torch.stack(x)) * multiplier
        else:
            raise ValueError("loss_aggregate_fn must be `sum` or `mean`")

        self.loss_aggregate_name = loss_aggregate_fn

    def _preprocess_loss(self, loss: Any) -> List[torch.Tensor]:
        if isinstance(loss, list):
            if self.loss_keys is not None:
                logger.warning(
                    f"Trying to get {self.loss_keys} keys from the losses, "
                    "but the loss is a list. All values will be aggregated."
                )
            result = loss
        elif isinstance(loss, dict):
            if self.loss_keys is not None:
                result = [loss[key] for key in self.loss_keys]
            else:
                result = list(loss.values())
        else:
            result = [loss]

        return result

    def on_batch_end(self, state: RunnerState) -> None:
        loss = state.get_key(key="loss")
        loss = self._preprocess_loss(loss)
        loss = self.loss_fn(loss)

        state.metrics.add_batch_value(
            metrics_dict={
                self.prefix: loss.item(),
            }
        )

        _add_loss_to_state(self.prefix, state, loss)
