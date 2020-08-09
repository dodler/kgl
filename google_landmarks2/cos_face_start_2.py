import glob
import logging
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm as tqdm
from sklearn import model_selection

from google_landmarks2.resnet import GeMPoolingLayer

tf.get_logger().setLevel(logging.ERROR)
import warnings

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    mixed_precision = True
else:
    mixed_precision = False

mixed_precision = False

if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")

config = {
    'learning_rate': 3e-3,
    'momentum': 0.9,
    'scale': 30,
    'margin': 0.3,
    'n_epochs': 240,
    'batch_size': 16,
    'input_size': (416, 416, 3),
    'n_classes': 81313,
    'dense_units': 512,
    'dropout_rate': 0.2,
}


def read_df(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)

    counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    df['probs'] = (
            (1 / df.counts ** alpha) / (1 / df.counts ** alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    uniques_map = dict(zip(uniques, range(len(uniques))))
    df['labels'] = df['landmark_id'].map(uniques_map)
    return df


df = read_df('/var/ssd_2t_1/kaggle_gld/')
print(df.shape)


def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):
    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst], axis=0), [3, 3])

    # convert degrees to radians
    rotation = math.pi * rotation / 360.
    shear = math.pi * shear / 360.

    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    rot_mat = get_3x3_mat([c1, s1, zero,
                           -s1, c1, zero,
                           zero, zero, one])

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_mat = get_3x3_mat([one, s2, zero,
                             zero, c2, zero,
                             zero, zero, one])

    zoom_mat = get_3x3_mat([one / hzoom, zero, zero,
                            zero, one / wzoom, zero,
                            zero, zero, one])

    shift_mat = get_3x3_mat([one, zero, hshift,
                             zero, one, wshift,
                             zero, zero, one])

    return tf.matmul(
        tf.matmul(rot_mat, shear_mat),
        tf.matmul(zoom_mat, shift_mat)
    )


def _spatial_transform(image,
                       rotation=3.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):
    ydim = tf.gather(tf.shape(image), 0)
    xdim = tf.gather(tf.shape(image), 1)
    xxdim = xdim % 2
    yxdim = ydim % 2

    # random rotation, shear, zoom and shift
    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    hzoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    wzoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

    # origin pixels
    y = tf.repeat(tf.range(ydim // 2, -ydim // 2, -1), xdim)
    x = tf.tile(tf.range(-xdim // 2, xdim // 2), [ydim])
    z = tf.ones([ydim * xdim], dtype='int32')
    idx = tf.stack([y, x, z])

    # destination pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    # clip to origin pixels range
    idx2y = tf.clip_by_value(idx2[0,], -ydim // 2 + yxdim + 1, ydim // 2)
    idx2x = tf.clip_by_value(idx2[1,], -xdim // 2 + xxdim + 1, xdim // 2)
    idx2 = tf.stack([idx2y, idx2x, idx2[2,]])

    # apply destinations pixels to image
    idx3 = tf.stack([ydim // 2 - idx2[0,], xdim // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    image = tf.reshape(d, [ydim, xdim, 3])
    return image


def _pixel_transform(image,
                     saturation_delta=0.3,
                     contrast_delta=0.1,
                     brightness_delta=0.2):
    image = tf.image.random_saturation(
        image, 1 - saturation_delta, 1 + saturation_delta)
    image = tf.image.random_contrast(
        image, 1 - contrast_delta, 1 + contrast_delta)
    image = tf.image.random_brightness(
        image, brightness_delta)
    return image


def preprocess_input(image, target_size, augment=False):
    image = tf.image.resize(
        image, target_size, method='bilinear')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image


def create_dataset(df, training, batch_size, input_size):
    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    def filter_by_probs(x, y, p):
        if p > np.random.uniform(0, 1):
            return True
        return False

    image_paths, labels, probs = df.path, df.labels, df.probs

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, probs))
    dataset = dataset.map(
        lambda x, y, p: (read_image(x), y, p),
        tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.filter(filter_by_probs)
    dataset = dataset.map(
        lambda x, y, p: (preprocess_input(x, input_size[:2], training), y),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


class AddMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin cosine distance.

    References:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''

    def __init__(self, n_classes, s=30, m=0.30, **kwargs):
        super(AddMarginProduct, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m

    def build(self, input_shape):
        super(AddMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        phi = cosine - self.m
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    References:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''

    def __init__(self, n_classes, s=30, m=0.30, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])
        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def create_model(input_shape,
                 n_classes,
                 dense_units=512,
                 dropout_rate=0.0,
                 scale=30,
                 margin=0.3):
    # backbone = tf.keras.applications.ResNet101V2(
    backbone = tf.keras.applications.EfficientNetB4(
        include_top=False,
        input_shape=input_shape,
        # weights='/home/lyan/Documents/kaggle/google_landmarks2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    )

    # pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')
    pooling = GeMPoolingLayer()
    batch_norm = tf.keras.layers.BatchNormalization(name='head/batch_norm')
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(dense_units, name='head/dense')

    margin = ArcMarginProduct(
        n_classes=n_classes,
        s=scale,
        m=margin,
        name='head/arc_margin',
        dtype='float32')

    softmax = tf.keras.layers.Softmax(dtype='float32')

    image = tf.keras.layers.Input(input_shape, name='input/image')
    label = tf.keras.layers.Input((), name='input/label')

    x = backbone(image)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    x = batch_norm(x)
    x = margin([x, label])
    x = softmax(x)
    return tf.keras.Model(
        inputs=[image, label], outputs=x)


class DistributedModel:

    def __init__(self,
                 input_size,
                 n_classes,
                 batch_size,
                 finetuned_weights,
                 dense_units,
                 dropout_rate,
                 scale,
                 margin,
                 optimizer,
                 strategy,
                 mixed_precision,
                 clip_grad=10.):

        self.model = create_model(
            input_shape=input_size,
            n_classes=n_classes,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            scale=scale,
            margin=margin, )

        self.input_size = input_size
        self.global_batch_size = batch_size * strategy.num_replicas_in_sync

        if finetuned_weights:
            self.model.load_weights(finetuned_weights)

        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy
        self.clip_grad = clip_grad

        # loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # metrics
        self.mean_loss_train = tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=False)
        self.mean_accuracy_train = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=1)
        self.mean_loss_valid = tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=False)
        self.mean_accuracy_valid = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=1)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, probs):
        per_example_loss = self.loss_object(labels, probs)
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=self.global_batch_size)

    def _backprop_loss(self, tape, loss, weights):
        gradients = tape.gradient(loss, weights)
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_grad)
        self.optimizer.apply_gradients(zip(clipped, weights))

    def _train_step(self, inputs):
        with tf.GradientTape() as tape:
            probs = self.model(inputs, training=True)
            loss = self._compute_loss(inputs[1], probs)
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)
        self._backprop_loss(tape, loss, self.model.trainable_weights)
        self.mean_loss_train.update_state(inputs[1], probs)
        self.mean_accuracy_train.update_state(inputs[1], probs)

    def _predict_step(self, inputs):
        probs = self.model(inputs, training=False)
        self.mean_loss_valid.update_state(inputs[1], probs)
        self.mean_accuracy_valid.update_state(inputs[1], probs)
        return probs

    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_replica_loss = self.strategy.run(self._train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def _distributed_predict_step(self, dist_inputs):
        probs = self.strategy.run(self._predict_step, args=(dist_inputs,))
        if tf.is_tensor(probs):
            return [probs]
        else:
            return probs.values

    def train_and_eval(self, train_ds, valid_ds, epochs, save_path):
        for epoch in range(epochs):
            dist_train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            dist_train_ds = tqdm.tqdm(dist_train_ds)
            for i, inputs in enumerate(dist_train_ds):
                loss = self._distributed_train_step(inputs)
                dist_train_ds.set_description(
                    "TRAIN: Loss {:.3f}, Accuracy {:.3f}".format(
                        self.mean_loss_train.result().numpy(),
                        self.mean_accuracy_train.result().numpy()
                    )
                )
            if valid_ds is not None:
                dist_valid_ds = self.strategy.experimental_distribute_dataset(valid_ds)
                dist_valid_ds = tqdm.tqdm(dist_valid_ds)
                for inputs in dist_valid_ds:
                    probs = self._distributed_predict_step(inputs)
                    dist_valid_ds.set_description(
                        "VALID: Loss {:.3f}, Accuracy {:.3f}".format(
                            self.mean_loss_valid.result().numpy(),
                            self.mean_accuracy_valid.result().numpy()
                        )
                    )

            if save_path:
                self.model.save_weights(save_path)

            self.mean_loss_train.reset_states()
            self.mean_loss_valid.reset_states()
            self.mean_accuracy_train.reset_states()
            self.mean_accuracy_valid.reset_states()


sss = model_selection.StratifiedShuffleSplit(
    n_splits=1, test_size=0.07, random_state=42
).split(X=df.index, y=df.landmark_id)

train_idx, valid_idx = next(sss)

train_ds = create_dataset(
    df=df.iloc[train_idx],
    training=True,
    batch_size=config['batch_size'],
    input_size=config['input_size'],
)
valid_ds = create_dataset(
    df=df.iloc[valid_idx],
    training=False,
    batch_size=config['batch_size'],
    input_size=config['input_size'],
)

with strategy.scope():
    optimizer = tf.keras.optimizers.SGD(
        config['learning_rate'], momentum=0.9, nesterov=True)

    dist_model = DistributedModel(
        input_size=config['input_size'],
        n_classes=config['n_classes'],
        batch_size=config['batch_size'],
        finetuned_weights=None,
        dense_units=config['dense_units'],
        dropout_rate=config['dropout_rate'],
        scale=config['scale'],
        margin=config['margin'],
        optimizer=optimizer,
        strategy=strategy,
        mixed_precision=True)

    dist_model.train_and_eval(
        train_ds=train_ds,
        # valid_ds=valid_ds,
        valid_ds=None,
        epochs=config['n_epochs'],
        save_path='eff_b4_arcface.h5')
