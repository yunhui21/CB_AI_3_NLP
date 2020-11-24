# Day_16_01_KerasCalifornia.py
from sklearn import datasets, model_selection, preprocessing
import tensorflow as tf
import numpy as np

# 문제
# 캘리포니아 주택 가격에 대해 60%로 학습하고 20%로 검증하고 나머지 20%에 대해 정확도를 구하세요


def california_regression_1():
    x, y = datasets.fetch_california_housing(return_X_y=True)
    print(x.shape, y.shape)  # (20640, 8) (20640,)

    x = preprocessing.scale(x)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.6)
    x_valid, x_test, y_valid, y_test = model_selection.train_test_split(x_test, y_test, train_size=0.5, shuffle=False)

    print(x_train.shape, x_valid.shape, x_test.shape)  # (12384, 8) (4128, 8) (4128, 8)
    print(y_train.shape, y_valid.shape, y_test.shape)  # (12384,) (4128,) (4128,)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse,
                  metrics=['mae'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
              validation_data=(x_valid, y_valid))
    print(model.evaluate(x_test, y_test))


def california_regression_2():
    x, y = datasets.fetch_california_housing(return_X_y=True)
    print(x.shape, y.shape)  # (20640, 8) (20640,)

    x = preprocessing.scale(x)

    # test_size = int(len(x) * 0.2)
    # valid_size = test_size
    # train_size = len(x) - test_size * 2
    #
    # x_train, y_train = x[:train_size], y[:train_size]
    # x_valid, y_valid = x[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
    # x_test, y_test = x[-test_size:], y[-test_size:]
    #
    # print(x_train.shape, x_valid.shape, x_test.shape)  # (12384, 8) (4128, 8) (4128, 8)
    # print(y_train.shape, y_valid.shape, y_test.shape)  # (12384,) (4128,) (4128,)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.8)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse,
                  metrics=['mae'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.3)
    print(model.evaluate(x_test, y_test))


# california_regression_1()
california_regression_2()
