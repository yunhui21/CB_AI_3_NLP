# Day_21_01_SoftmaxRegresssion_iris.py
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import tensorflow as tf


# 문제 4
# 30%의 데이터에 대해 정확도를 구하세요
def show_accuracy(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == y_arg)
    print('acc :', np.mean(equals))


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


# 문제
# iris 파일을 읽어서 (iris(150).csv)
# 70%로 학습하고 30%에 대해 정확도를 구하세요
def softmax_iris():
    # 문제 1
    # 붓꽃 데이터 파일을 읽어오세요
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    print(iris)

    # 문제 2
    # x, y 데이터로 변환하세요
    # x = iris.values[:, :-1]
    # y = iris.values[:, -1]

    enc = preprocessing.LabelBinarizer()
    y = enc.fit_transform(iris.Species)

    # df = iris.drop(['Species'], axis=1)
    # x = df.values

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values

    x = np.float32(x)

    print(x.shape, y.shape)     # (150, 4) (150, 3)
    print(y[:3])
    print(y[-3:])

    # 문제 3
    # 70%의 데이터로 학습하세요
    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    n_features = x.shape[1]
    n_classes = y.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    # (150, 3) = (150, 4) @ (4, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    # 문제
    # show_accuracy 함수를 호출하세요
    preds_train = sess.run(hx, {ph_x: x_train})
    preds_test = sess.run(hx, {ph_x: x_test})

    show_accuracy(preds_train, y_train)
    show_accuracy(preds_test, y_test)

    sess.close()


def softmax_iris_sparse():
    # 문제 1
    # 붓꽃 데이터 파일을 읽어오세요
    iris = pd.read_csv('data/iris(150).csv', index_col=0)
    print(iris)

    # 문제 2
    # x, y 데이터로 변환하세요
    # x = iris.values[:, :-1]
    # y = iris.values[:, -1]

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(iris.Species)

    # 원핫 벡터 인코딩
    # y = np.eye(3, dtype=np.int32)[y]

    # df = iris.drop(['Species'], axis=1)
    # x = df.values

    iris.drop(['Species'], axis=1, inplace=True)
    x = iris.values

    x = np.float32(x)

    print(x.shape, y.shape)     # (150, 4) (150,)
    print(y[:3])
    print(y[-3:])

    # 문제 3
    # 70%의 데이터로 학습하세요
    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data
    print(y_train.shape, y_test.shape)      # (105,) (45,)

    n_features = x.shape[1]
    n_classes = 3
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    # (150, 3) = (150, 4) @ (4, 3)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    # 문제
    # show_accuracy 함수를 호출하세요
    preds_train = sess.run(hx, {ph_x: x_train})
    preds_test = sess.run(hx, {ph_x: x_test})

    show_accuracy_sparse(preds_train, y_train)
    show_accuracy_sparse(preds_test, y_test)

    sess.close()


softmax_iris()
# softmax_iris_sparse()



