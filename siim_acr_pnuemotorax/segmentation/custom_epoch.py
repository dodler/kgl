import sys
import torch
from tqdm import tqdm as tqdm
from torchnet.meter import AverageValueMeter
from tensorboardX import SummaryWriter
import datetime


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        # self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class ExperimentWriter:
    def __init__(self, experiment_name=None):
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if experiment_name is None:
            sw = SummaryWriter(log_dir='/tmp/runs/' + dt)
        else:
            sw = SummaryWriter(log_dir='/tmp/runs/' + experiment_name)
        self.sw = sw

    def write_image_to_tensorboard(self, x, y, prediction, prefix='train_image'):
        pred = prediction > 0.5
        pred = pred.long()
        self.sw.add_image(prefix + '/image', x[0, :, :, :])
        self.sw.add_image(prefix + '/gt', y[0, :, :, :])
        self.sw.add_image(prefix + '/pred@0.5', pred[0, :, :, :])

    def write_loss(self, loss_val, loss_tag, cnt):
        self.sw.add_scalar(str(loss_tag) + '/loss', loss_val, cnt)


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, experiment_name, device='cpu', opt_step_size=1, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.exp_writer=ExperimentWriter(experiment_name)
        self.opt_step_size = opt_step_size
        self.optimizer = optimizer
        self.cnt = 0
        self.loss_batch = 0

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.cnt += 1
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.loss_batch += loss
        self.exp_writer.write_loss(loss.detach().cpu().item(), "train", self.cnt)

        if self.cnt % 100 == 0:
            self.exp_writer.write_image_to_tensorboard(x.detach().cpu(), y.detach().cpu().numpy(), prediction.detach().cpu())

        if self.cnt % self.opt_step_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.loss_batch = 0

        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, experiment_name, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.exp_writer=ExperimentWriter(experiment_name)
        self.cnt = 0

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        self.cnt += 1
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        self.exp_writer.write_loss(loss.detach().cpu().item(), "valid", self.cnt)
        self.exp_writer.write_image_to_tensorboard(x.detach().cpu(),
                                   y.detach().cpu().numpy(),
                                   prediction.detach().cpu(),
                                   prefix='valid_images')
        return loss, prediction
