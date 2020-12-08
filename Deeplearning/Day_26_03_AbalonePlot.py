# Day_26_03_AbalonePlot.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)

# 문제 1
# 26-2 파일의 코드를 멀티플 리그레션으로 수정하세요

# 문제 2
# 10 에포크마다 발생하는 손실과 오차를 그래프로 그리세요


def get_data():
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
    abalone = pd.read_csv('data/abalone.data', header=None, names=names)

    enc = preprocessing.LabelEncoder()
    gender = enc.fit_transform(abalone.Sex)
    y = abalone.Rings.values
    y = y.reshape(-1, 1)

    abalone.drop(['Sex', 'Rings'], axis=1, inplace=True)
    x = np.hstack([gender.reshape(-1, 1), abalone.values])      # (4177, 8) (4177,)

    print(x.shape, y.shape)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    ytest_1 = labels.reshape(-1)

    diff = preds_1 - ytest_1
    diff_abs = np.abs(diff)

    avg = np.mean(diff_abs)
    # print('오차 평균 :', avg)

    return avg


def model_abalone_regression(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    losses, errors = [], []
    for i in range(500):
        sess.run(train, {ph_x: x_train})
        c = sess.run(loss, {ph_x: x_train})
        losses.append(c)

        preds_test = sess.run(hx, {ph_x: x_test})
        avg = show_difference(preds_test, y_test)
        errors.append(avg)

    sess.close()

    indices = range(len(errors))
    plt.plot(indices, errors, label='error')
    plt.plot(indices, losses, label='loss')
    plt.legend()
    plt.show()


x_train, x_test, y_train, y_test = get_data()
model_abalone_regression(x_train, x_test, y_train, y_test)









