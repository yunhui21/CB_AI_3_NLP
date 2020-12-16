# Day_32_02_callback.py
# uci machine learning car evaluation
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
# 문제
# 자동차 데이터에 대해 80%로 학습하고 20%에 대해 정확도를 구하세요.


# dense  : LabelBinarizer
# sparse : LabelEncoder
def get_cars_sparse():
    names =  ['buying', 'maint', 'doors', 'person', 'lug_boot', 'safety','acc']
    cars = pd.read_csv('data/car.data', header = None, names = names)
    # print(cars)

    # enc = preprocessing.LabelEncoder()
    # encoder 1차원
    enc = preprocessing.LabelBinarizer()
    # binarizer 2차원

    buying   = enc.fit_transform(cars.buying)
    maint    = enc.fit_transform(cars.maint)
    doors    = enc.fit_transform(cars.doors)
    person   = enc.fit_transform(cars.person)
    lug_boot = enc.fit_transform(cars.lug_boot)
    safety   = enc.fit_transform(cars.safety)
    acc      = enc.fit_transform(cars.acc)

    # print(buying.shape, maint.shape) # (1728,) (1728,)
    # data = np.transpose([buying, maint, doors, person, lug_boot, safety])
    data = np.hstack([buying, maint, doors, person, lug_boot, safety ])
    acc = np.argmax(acc, axis=1)
    print(data.shape) # (1728, 7)

    #--------------------------------------------------------#
    train_size = int(len(data) * 0.8)

    ds_train = tf.data.Dataset.from_tensor_slices(data[:train_size])
    ds_train = ds_train.map(lambda chunk: (chunk[:-1], chunk[-1]))  # 2개로 나눠주는 역할.x (32, 10)
    ds_train = ds_train.batch(32, drop_remainder=True)  # 104


    ds_test = tf.data.Dataset.from_tensor_slices(data[train_size:])
    ds_test = ds_test.map(lambda chunk: (chunk[:-1], chunk[-1]))
    ds_test = ds_test.batch(32, drop_remainder=True)

    return ds_train, ds_test

def build_model(n_classes):
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(60, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    return model

class EveryBatch(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_loss = []
        self.batch_acc  = []

    # def on_batch_begin(self, batch, logs=None):
    #     print('on_batch_begin', batch, logs)
    #
    # def on_batch_end(self, batch, logs=None):
    #     print('on_batch_end', batch, logs)
    #
    #     self.batch_loss.append(logs['loss'])
    #     self.batch_acc.append(logs['acc'])

    # val_loss가 나오는 것이 아니라 on_batch_end 함수와 동일하게 loss가 나온다.
    def on_test_batch_end(self, batch, logs=None):
         print('on_test_batch_end', logs)

ds_train, ds_test = get_cars_sparse()
model = build_model(4)

# checkpoint = tf.keras.callbacks.ModelCheckpoint('model_callback/first.h5')
# checkpoint = tf.keras.callbacks.ModelCheckpoint('model_callback/keras_{epoch:03d}_val_loss_{val_loss:.4f}.h5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model_callback/keras_val_acc_{epoch:03d}_{val_acc:.4f}.h5',
    monitor='val_acc',
    save_best_only=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=5
)

model.fit(ds_train, epochs=30, verbose=2,
          validation_data = ds_test,
          callbacks=[checkpoint, early_stopping])
# print('acc:', model.evaluate(ds_test))
