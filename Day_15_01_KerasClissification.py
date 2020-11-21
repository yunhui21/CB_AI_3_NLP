# Day_15_01_KerasClissification.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 문제
# 피마 인디언 파일을 읽어서 79%로 학습하고 30%에 대해 정확도를 구하세요.

# 문제
# iris 파일을 읽어서 70%로 학습하고 30%에 대해 정확도를 구하세요..


def logistic_regression():
    pass

def logistic_regression_pima_1():

    pima = pd.read_csv('data/pima-indians.csv')
    print(pima)

    x = pima.values[:, :-1]
    # y = pima.values[:, -1:]
    y = pima.Outcome.values
    y = y.reshape(-1, 1)

    print(x.shape, y.shape)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.7)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
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

    print(x.shape, y.shape)

    x = preprocessing.scale(x)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.7)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train,
              epochs=100,
              batch_size=32,
              verbose=2,
              validation_data=(x_test, y_test))         # 반드시 튜플로 전달, 리스트 안됨.
    # print(model.evaluate(x_test, y_test, verbose=0))


def logistic_regression_pima_3():
    pima = pd.read_csv('data/pima-indians.csv')
    print(pima)

    x = pima.values[:, :-1]
    # y = pima.values[:, -1:]
    y = pima.Outcome.values
    y = y.reshape(-1, 1)

    print(x.shape, y.shape)

    x = preprocessing.scale(x)

    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = 0.7)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y,
              epochs=100,
              batch_size=32,
              verbose=2,
              validation_split=0.3)         # 반드시 튜플로 전달, 리스트 안됨.
              # 데이터를 바꾼다. 매번 데이터를 바꾼다. epochs값..데이터 shffle..
    # print(model.evaluate(x_test, y_test, verbose=0))


def softmax_regression():
    x = [[1, 2],
         [2, 1],
         [4, 5],
         [5, 4],
         [8, 9],
         [9, 8]]
    y = [[0, 0, 0],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x,y,verbose=0))

    preds = model.predict(x)
    print(preds)

    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(preds_arg)
    print(y_arg)

    print('acc:', np.mean(preds_arg == y_arg))


def softmax_regression_iris():
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    # print(iris) # [150 rows x 5 columns]

    x = iris.values[:, :-1]
    y = preprocessing.LabelBinarizer().fit_transform(iris.Species)
    # print(x.shape, y.shape) # (150, 4) (150,)
    # print(y[:3]) #[[1 0 0] [1 0 0] [1 0 0]]
    # print(x.dtype)  # float32
    # print(x[:3]) # # [[5.1 3.5 1.4 0.2][4.9 3.  1.4 0.2][4.7 3.2 1.3 0.2]]

    # x = preprocessing.scale(x)
    x = np.float32(x)
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2,
              validation_split=0.3)
    # print(model.evaluate(x_test,y_test,verbose=0))
    #
    # preds = model.predict(x)
    # print(preds)
    #
    # preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(y, axis=1)
    # print(preds_arg)
    # print(y_arg)
    #
    # print('acc:', np.mean(preds_arg == y_arg))


# logistic_regression()
# logistic_regression_pima_1()
# logistic_regression_pima_2()
# softmax_regression()
softmax_regression_iris()