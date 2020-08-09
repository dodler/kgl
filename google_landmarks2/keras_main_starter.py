import pickle

import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, SGD
from numba import jit
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

from google_landmarks2 import make_one_hot
from google_landmarks2.resnet import ResNet50, pretrained_r50
from google_landmarks2.resnet2 import DelfArcFaceModel
from google_landmarks2.sched import CyclicLR, StepDecay


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


import os
import tensorflow as tf
import skimage.io
from skimage.transform import resize
from sklearn import preprocessing
# from tensorflow.keras.applications.resnet50 import ResNet50
from tqdm import tqdm
from imgaug import augmenters as iaa
import keras.backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config));

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

train = pd.read_csv("/var/ssd_2t_1/kaggle_gld/train.csv")


def get_paths(sub):
    index = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]

    paths = []

    for a in index:
        for b in index:
            for c in index:
                try:
                    paths.extend([f"/var/ssd_2t_1/kaggle_gld/train/{sub}/{a}/{b}/{c}/" + x for x in
                                  os.listdir(f"/var/ssd_2t_1/kaggle_gld/train/{sub}/{a}/{b}/{c}")])
                except:
                    pass

    return paths


train_path = train

if os.path.exists('rows.pkl'):
    with open('rows.pkl', 'rb') as f:
        rows = pickle.load(f)
else:
    print('collecting paths to train')
    rows = []
    for i in tqdm(range(len(train))):
        row = train.iloc[i]
        path = list(row["id"])[:3]
        temp = row["id"]
        row["id"] = f"/var/ssd_2t_1/kaggle_gld/train/{path[0]}/{path[1]}/{path[2]}/{temp}.jpg"
        rows.append(row["id"])

    with open('rows.pkl', 'wb') as f:
        pickle.dump(rows, f)

rows = pd.DataFrame(rows)
train_path["id"] = rows
batch_size = 32
seed = 42
shape = (224, 224, 3)  ##desired shape of the image for resizing purposes
val_sample = 0.1  # 10 % as validation sample

train_labels = pd.read_csv('/var/ssd_2t_1/kaggle_gld/train.csv')
train_labels.head()
k = train[['id', 'landmark_id']].groupby(['landmark_id']).agg({'id': 'count'})
k.rename(columns={'id': 'Count_class'}, inplace=True)
k.reset_index(level=(0), inplace=True)
freq_ct_df = pd.DataFrame(k)
freq_ct_df.head()
train_labels = pd.merge(train, freq_ct_df, on=['landmark_id'], how='left')
train_labels.head()
freq_ct_df.sort_values(by=['Count_class'], ascending=False, inplace=True)
freq_ct_df.head()

n_top_class = 81313
# n_top_class = 10000
print(freq_ct_df.shape)
freq_ct_df_top100 = freq_ct_df.iloc[:n_top_class]
top100_class = freq_ct_df_top100['landmark_id'].tolist()
top100class_train = train_path[train_path['landmark_id'].isin(top100_class)]
print(top100class_train.shape)


def getTrainParams():
    data = top100class_train.copy()
    le = preprocessing.LabelEncoder()
    data['label'] = le.fit_transform(data['landmark_id'])
    lbls = top100class_train['landmark_id'].tolist()
    # lb = LabelBinarizer()
    # labels = lb.fit_transform(lbls)

    # return np.array(top100class_train['id'].tolist()), np.array(labels), le
    return np.array(top100class_train['id'].tolist()), np.array(lbls), le


class Landmark2020_DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, shape, n=1000, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
        self.n = n

    def __len__(self):
        if self.n == -1:
            return int(np.ceil(len(self.paths) / float(self.batch_size)))
        else:
            return self.n

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
        y = make_one_hot(y, n_samples=y.shape[0], n_class=n_top_class)

        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),  # horizontal flips

                    iaa.ContrastNormalization((0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),

                    iaa.Affine(rotate=0),
                    # iaa.Affine(rotate=90),
                    # iaa.Affine(rotate=180),
                    # iaa.Affine(rotate=270),
                    iaa.Fliplr(0.5),
                    # iaa.Flipud(0.5),
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)

        # print('data read ok')
        return [X, y], y

    def on_epoch_end(self):

        # indices = np.arange(0, len(self.paths))
        # indices = np.random.choice(indices, self.n, replace=False)
        # self.indices = indices

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        # image_norm = skimage.io.imread(path) / 255.0
        image_norm = cv2.imread(path)
        image_norm = cv2.cvtColor(image_norm, cv2.COLOR_BGR2RGB) / 255.0

        # im = resize(image_norm, (shape[0], shape[1], shape[2]), mode='reflect')
        im = cv2.resize(image_norm, (224, 224))
        return im


def create_model(input_shape, n_out):
    print('create model using n_out', n_out)
    # input = Input(input_shape)

    # weight_decay = 1e-4

    # model = ResNet50(input_shape=input_shape, n_out=n_out, embed=2048)
    model = pretrained_r50(input_shape=input_shape, n_out=n_out, embed=2048)

    # model = DelfArcFaceModel(input_shape=input_shape,
    #                          n_classes=n_out,
    #                          margin=0.1,
    #                          logit_scale=1,
    #                          feature_size=2048)
    # model.build(input_shape=input_shape)

    model.compile(loss='categorical_crossentropy',
                  # optimizer=Adam(lr=5e-3, decay=5e-4),
                  optimizer=SGD(lr=1e-3, momentum=0.9, decay=5e-4),
                  metrics=['accuracy'])
    # model.load_weights('model.hdf5')

    return model


nlabls = top100class_train['landmark_id'].nunique()
print('num labels', nlabls)
model = create_model(input_shape=(224, 224, 3), n_out=nlabls)
model.summary()
paths, labels, _ = getTrainParams()
keys = np.arange(paths.shape[0], dtype=np.int)
np.random.seed(seed)
np.random.shuffle(keys)
lastTrainIndex = int((1 - val_sample) * paths.shape[0])

### debug
# ----------------------
if False:
    b_size = 1

    t = labels[0:5].copy()
    t[0] = 0
    t[1] = 0
    t[2] = 1
    t[3] = 1
    t[4] = 2

    y = make_one_hot(t, n_samples=t.shape[0], n_class=3)
    print('labels', t)
    print('my bin', y.shape, y)

    y_ = LabelBinarizer().fit_transform(t)
    print('label bin', y_.shape, y_)

    raise Exception()

# ---------------------

if __name__ == '__main__':

    if False:
        ### debug
        for path in tqdm(paths):
            image_norm = skimage.io.imread(path) / 255.0
            # im = resize(image_norm, (shape[0], shape[1], shape[2]), mode='reflect')

    unique_labels = np.unique(labels)
    labels_map = {}
    for i in range(n_top_class):
        labels_gt = unique_labels[i]
        labels_map[labels_gt] = i
    for i in range(labels.shape[0]):
        labels[i] = labels_map[labels[i]]

    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]

    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]

    print(paths.shape, labels.shape)
    print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
    train_generator = Landmark2020_DataGenerator(pathsTrain, labelsTrain, batch_size,
                                                 shape, use_cache=False, n=4000,
                                                 augment=False,
                                                 shuffle=True)
    val_generator = Landmark2020_DataGenerator(pathsVal, labelsVal, batch_size, shape, n=-1,
                                               use_cache=False, shuffle=False)
    # clr = CyclicLR(base_lr=0.0005, max_lr=0.01, step_size=2000., mode='triangular')
    sched = LearningRateScheduler(StepDecay(dropEvery=15, initAlpha=1e-2, factor=0.1))
    epochs = 50
    use_multiprocessing = False
    workers = 6
    base_cnn = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=100,
        # class_weight = class_weights,
        epochs=epochs,
        callbacks=[
            ModelCheckpoint('model.hdf5', verbose=1, save_best_only=True),
            sched
        ],
        use_multiprocessing=use_multiprocessing,
        workers=workers,
        max_queue_size=10,
        verbose=1)
    model.save('ResNet50.h5')
