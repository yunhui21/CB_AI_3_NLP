# Day_32_01_01_tfdata.py
#Day_31_01 SaveModel
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection


# 이전 파일에서 가져온 함수
def get_abalone():
    names = ('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
    abalone = pd.read_csv('data/abalone.data', header=None, names=names)
    # print(abalone)

    # y = []
    # for r in abalone.rings:
    #     if   r <= 8 : y.append(0)
    #     elif r <= 10: y.append(1)
    #     else        : y.append(2)
    # print(y[:10])   # [2, 0, 1, 1, 0, 0, 2, 2, 1, 2]

    #                   1                    8     10
    # categories = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    categories = [-1] + [0] * 8 + [1] * 2 + [2] * 19
    categories = np.int32(categories)

    y = categories[abalone.rings]
    # print(y[:10])     # [2 0 1 1 0 0 2 2 1 2]

    # x = abalone.values[:, 1:-1]
    # print(x.shape, x.dtype) # (4177, 7) object
    # x = np.float32(x)

    x = abalone.drop(['sex', 'rings'], axis=1).values
    # print(x.shape, x.dtype) # (4177, 7) float64

    sex = preprocessing.LabelBinarizer().fit_transform(abalone.sex)
    # print(sex[:3])  # [[0 0 1] [0 0 1] [1 0 0]]

    # x = np.concatenate([sex, x], axis=1)
    x = np.hstack([sex, x])
    x = preprocessing.scale(x)
    # print(x.shape)       # (4177, 10)

    return model_selection.train_test_split(x, y, train_size=0.8)

# 문제
# 학습과 검사에 사용할 tf.data.Dataset 객체를 반환하는 함수를 만드세요.
def get_abalone_tfdata():
    names = (
    'sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
    abalone = pd.read_csv('data/abalone.data', header=None, names=names)

    categories = [-1] + [0] * 8 + [1] * 2 + [2] * 19
    categories = np.int32(categories)

    y = categories[abalone.rings]
    x = abalone.drop(['sex', 'rings'], axis=1).values

    sex = preprocessing.LabelBinarizer().fit_transform(abalone.sex)

    x = np.hstack([sex, x])
    x = preprocessing.scale(x)

    data = np.hstack([x, y.reshape(-1, 1)])
    # print(data.shape) # (4177, 11)

    train_size = int(len(data)*0.8)
    # 가장 좋은 코드이지만, map 함수를 사용하고 싶어서 사용하지 않음.
    # ds_train = tf.data.Dataset.from_tensor_slices((x,y))


    # ds_train = tf.data.Dataset.from_tensor_slices(data[:train_size])
    # # for take in ds_train.take(2):
    # #     print(take.numpy())
    # # [-0.67483383 - 0.68801788  1.31667716 - 0.57455813 - 0.43214879 - 1.06442415
    # #  - 0.64189823 - 0.60768536 - 0.72621157 - 0.63821689  2.]
    # ds_train = ds_train.map(lambda chunk: (chunk[:-1], chunk[-1]))
    # # for xx, yy in ds_train.take(2):
    # #     print(xx.numpy(), yy.numpy())
    #     # [-0.67483383 -0.68801788  1.31667716 -0.57455813 -0.43214879 -1.06442415
    #     #  -0.64189823 -0.60768536 -0.72621157 -0.63821689] 2.0
    #
    # ds_train = ds_train.batch(32, drop_remainder=True)  # 104
#
    ds_train = tf.data.Dataset.from_tensor_slices(data[:train_size])
    ds_train = ds_train.batch(32, drop_remainder=True)  # 104
    # for take in ds_train.take(2):
    #     print(take.shape) # (32, 11)
    ds_train = ds_train.map(lambda chunk: (chunk[:, :-1], chunk[:, -1])) # 2개로 나눠주는 역할.x (32, 10)
    # for xx, yy in ds_train.take(2):
    #     print(xx.shape, yy.shape) # (32, 10) (32,)

    ds_test = tf.data.Dataset.from_tensor_slices(data[train_size:])
    ds_test = ds_test.map(lambda chunk: (chunk[:-1], chunk[-1]))
    ds_test = ds_test.batch(32, drop_remainder=True)

    return ds_train, ds_test

def build_model(n_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(60, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    return model

# map 함수 사용 안함.
def model_abalone_bad():
    x_train, x_test, y_train, y_test = get_abalone()
    model = build_model(n_classes = len(set(y_train)))


    # 문제
    # x, y로 나누어진 데이터를 tf.data.Datasets 클래스로 변환하세요.
    # 변환 데이터를 fit 함수에 넣어서 결과가 나오게 해주세요.
    # (Day_28_02_chosun.py파일에 있는 model_chosun_2 함수 참고)

    # 입력          출력
    # (4177,)    -> () scalar
    # (4177, 10) -> (10,)
    # (3, 4, 12) -> (4, 12)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) # []전달불가

    # for take in ds_train.take(2):
    #     print(type(take), len(take)) # <class 'tuple'> 2

    # for xx, yy in zip(x_train, y_train):
    #     print(xx, yy)

    # for xx, yy in ds_train.take(2):
    #     print(xx.numpy(), yy.numpy())

    ds_train = ds_train.shuffle(buffer_size=len(x_train))   # 별 의미 없다. 공부차원에서 시도.
    ds_train = ds_train.batch(32)

    # model.fit(ds_train, epochs=10, verbose=2)
    # Epoch 10/10
    # 105/105 - 0s - loss: 0.7357 - acc: 0.6618

    # repeat 함수를 학습에 적용하는 것은 맞지 않다.
    # model.fit(ds_train.repeat(), epochs=10, verbose=2) # error
    model.fit(ds_train.repeat(), epochs=10, verbose=2, steps_per_epoch=210) # error
    # Epoch 10/10
    # 210/210 - 0s - loss: 0.7132 - acc: 0.6661

    # model.fit(ds_train.repeat(5), epochs=10, verbose=2)
    # Epoch 10/10
    # 525/525 - 0s - loss: 0.6952 - acc: 0.6778

    # model.fit(ds_train.repeat(1), epochs=10, verbose=2)
    # Epoch 10/10
    # 105/105 - 0s - loss: 0.7282 - acc: 0.6645



    # print('acc:', model.evaluate(x_test, y_test))

# map 함수 사용
def model_abalone_good():
    ds_train, ds_test = get_abalone_tfdata()
    model = build_model(3)

    model.fit(ds_train, epochs=50, verbose=2)
    print('acc:', model.evaluate(ds_test))

# model_abalone_bad()
model_abalone_good()

