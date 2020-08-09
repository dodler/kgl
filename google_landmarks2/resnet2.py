import tensorflow as tf
import functools
import math


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False):
        super().__init__()
        if train_p:
            self.p = tf.Variable(p, dtype=tf.float32)
        else:
            self.p = p
        self.eps = 1e-6

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs


class DelfArcFaceModel(tf.keras.Model):
    def __init__(self, input_shape, n_classes, margin, logit_scale, feature_size, p=None, train_p=False):
        super().__init__()
        self.backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        self.backbone.summary()

        if p is not None:
            self.global_pooling = GeMPoolingLayer(p, train_p=train_p)
        else:
            self.global_pooling = functools.partial(tf.reduce_mean, axis=[1, 2], keepdims=False)
        self.dense1 = tf.keras.layers.Dense(feature_size, activation=None, kernel_initializer="glorot_normal")
        # self.bn1 = tf.keras.layers.BatchNormalization()
        self.arcface = ArcFaceLayer(n_classes, margin, logit_scale)

    def call(self, inputs, training=True, mask=None):
        images, labels = inputs
        x = self.extract_feature(images)
        x = self.arcface((x, labels))
        return x

    def extract_feature(self, inputs):
        x = self.backbone(inputs)
        x = self.global_pooling(x)
        x = self.dense1(x)
        return x


class ArcFaceLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin, logit_scale):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.logit_scale = logit_scale

    def build(self, input_shape):
        self.w = self.add_weight("weights", shape=[int(input_shape[0][-1]), self.num_classes], initializer=tf.keras.initializers.get("glorot_normal"))
        self.cos_m = tf.identity(tf.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(tf.sin(self.margin), name='sin_m')
        self.th = tf.identity(tf.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, inputs, training=True, mask=None):
        embeddings, labels = inputs
        normed_embeddings = tf.nn.l2_normalize(embeddings, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embeddings, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = tf.multiply(logits, self.logit_scale, 'arcface_logist')
        return logits