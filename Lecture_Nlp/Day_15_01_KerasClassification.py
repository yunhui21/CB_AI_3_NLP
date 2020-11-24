# Day_15_01_KerasClassification.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 문제
# 피마 인디언 파일을 읽어서 70%로 학습하고 30%에 대해 정확도를 구하세요

# 문제
# iris 파일을 읽어서 70%로 학습하고 30%에 대해 정확도를 구하세요


def logistic_regression():
    x = [[1, 2],        # fail
         [2, 1],
         [4, 5],        # pass
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[0],
         [0],
         [1],
         [1],
         [1],
         [1]]

    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
    # model.add(tf.keras.layers.Dense(1, activation=tf.sigmoid))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # z = tf.matmul(x, w) + b
    # hx = tf.nn.sigmoid(z)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    preds = model.predict(x)
    print(preds)

    preds_bool = (preds > 0.5)
    print(preds_bool)

    equals = (preds_bool == y)
    print(equals)
    print('acc :', np.mean(equals))


def logistic_regression_pima_1():
    pima = pd.read_csv('data/pima-indians.csv')
    print(pima)

    x = pima.values[:, :-1]
    # y = pima.values[:, -1:]
    y = pima.Outcome.values
    y = y.reshape(-1, 1)

    print(x.shape, y.shape)             # (768, 8) (768, 1)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)
    print(x_train.shape, x_test.shape)  # (537, 8) (231, 8)
    print(y_train.shape, y_test.shape)  # (537, 1) (231, 1)

    # -------------------------------------- #

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


def logistic_regression_pima_2():
    pima = pd.read_csv('data/pima-indians.csv')
    print(pima)

    x = pima.values[:, :-1]
    # y = pima.values[:, -1:]
    y = pima.Outcome.values
    y = y.reshape(-1, 1)

    print(x.shape, y.shape)             # (768, 8) (768, 1)

    x = preprocessing.scale(x)          # 표준화

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)
    print(x_train.shape, x_test.shape)  # (537, 8) (231, 8)
    print(y_train.shape, y_test.shape)  # (537, 1) (231, 1)

    # -------------------------------------- #

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train,
              epochs=100, batch_size=32, verbose=2,
              validation_data=(x_test, y_test))         # 반드시 튜플로 전달. 리스트 안됨
    # print(model.evaluate(x_test, y_test, verbose=0))


def logistic_regression_pima_3():
    pima = pd.read_csv('data/pima-indians.csv')
    print(pima)

    x = pima.values[:, :-1]
    # y = pima.values[:, -1:]
    y = pima.Outcome.values
    y = y.reshape(-1, 1)

    print(x.shape, y.shape)             # (768, 8) (768, 1)

    x = preprocessing.scale(x)          # 표준화

    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)
    # print(x_train.shape, x_test.shape)  # (537, 8) (231, 8)
    # print(y_train.shape, y_test.shape)  # (537, 1) (231, 1)

    # -------------------------------------- #

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

#     model.fit(x_train, y_train,
#               epochs=100, batch_size=32, verbose=2,
#               validation_data=(x_test, y_test))         # 반드시 튜플로 전달. 리스트 안됨

    model.fit(x, y,
              epochs=100, batch_size=32, verbose=2,
              validation_split=0.3)
    # print(model.evaluate(x_test, y_test, verbose=0))


def softmax_regression():
    x = [[1, 2],        # C
         [2, 1],
         [4, 5],        # B
         [5, 4],
         [8, 9],        # A
         [9, 8]]
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # z = tf.matmul(x, w) + b
    # hx = tf.nn.softmax(z)

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    preds = model.predict(x)
    print(preds)

    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(preds_arg)
    print(y_arg)

    print('acc :', np.mean(preds_arg == y_arg))


def softmax_regression_iris():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    print(iris)

    # validation_split을 사용할 때, 그룹화된 배열을 섞지 않으면 결과 안 나옴.
    iris = iris.values
    np.random.shuffle(iris)

    x = iris[:, :-1]
    y = preprocessing.LabelEncoder().fit_transform(iris[:, -1])
    print(x.shape, y.shape)     # (150, 4) (150,)
    print(x.dtype, y.dtype)     # object int64

    x = preprocessing.scale(x)  # float64
    # x = np.float32(x)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    # 문제
    # validation_split 옵션에 0.3을 주었을 때, 제대로 된 정확도가 나오도록 수정하세요
    model.fit(x, y, epochs=100, verbose=2, validation_split=0.3)
    # print(model.evaluate(x, y, verbose=0))


# logistic_regression()
# logistic_regression_pima_1()
# logistic_regression_pima_2()
# logistic_regression_pima_3()

# softmax_regression()
softmax_regression_iris()
