import sys
import torch
from tqdm import tqdm as tqdm
from torchnet.meter import AverageValueMeter
from tensorboardX import SummaryWriter
import datetime


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


global exp_writer
exp_writer = None


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, experiment_name=''):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        global exp_writer
        exp_writer = ExperimentWriter(experiment_name=experiment_name)
        self.cnt = 0

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

        global exp_writer

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                if self.cnt % 30 == 0:
                    exp_writer.write_image_to_tensorboard(x.detach().cpu(), y.detach().cpu().numpy(),
                                                               y_pred.detach().cpu())

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                exp_writer.sw.add_scalar(self.stage_name+'/loss_'+self.loss.__name__, loss_value, self.cnt)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()

                    exp_writer.sw.add_scalar(self.stage_name+'/metric_'+metric_fn.__name__, metric_value, self.cnt)

                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                self.cnt += 1

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, experiment_name, device='cpu', opt_step_size=1,
                 verbose=True, enorm=None):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            experiment_name=experiment_name,
        )
        self.enorm = enorm
        self.opt_step_size = opt_step_size
        self.optimizer = optimizer
        self.loss_batch = 0

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.loss_batch += loss

        if self.cnt % self.opt_step_size == 0:
            self.optimizer.step()
            if self.enorm is not None:
                self.enorm.step()
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
            experiment_name=experiment_name
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
