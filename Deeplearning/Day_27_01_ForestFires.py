# Day_27_01_ForestFires.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)


# 문제 1
# 산불 데이터에 대해 오차를 구하세요 (area)

# 문제 2
# loss와 결과를 서로 다른 그래프에 그리세요
# 각각 train과 test 두 개의 그래프로 이루어져야 합니다


def get_data():
    fire = pd.read_csv('data/forestfires.csv')
    print(fire)
    fire.info()
    print(fire.describe())

    enc = preprocessing.LabelEncoder()
    month = enc.fit_transform(fire.month)
    month = month.reshape(-1, 1)

    day = enc.fit_transform(fire.day)
    day = day.reshape(-1, 1)

    y = fire.area.values
    y = y.reshape(-1, 1)

    fire.drop(['month', 'day', 'area'], axis=1, inplace=True)
    x = np.hstack([month, day, fire.values])

    print(x.shape, y.shape)         # (517, 12) (517, 1)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    ytest_1 = labels.reshape(-1)

    diff = preds_1 - ytest_1
    diff_abs = np.abs(diff)

    avg = np.mean(diff_abs)
    # print('오차 평균 :', avg)

    return avg


def show_plot(idx, caption, values1, title1, values2, title2):
    plt.subplot(1, 2, idx)
    indices = range(len(values1))
    plt.plot(indices, values1, label=title1)
    plt.plot(indices, values2, label=title2)
    plt.legend()
    plt.title(caption)


def model_forest_fires_regression(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    hx = tf.matmul(ph_x, w) + b

    loss_i = (hx - ph_y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    dic_train = {ph_x: x_train, ph_y: y_train}
    dic_test = {ph_x: x_test, ph_y: y_test}

    c_trains, c_tests, avg_trains, avg_tests = [], [], [], []
    for i in range(100):
        sess.run(train, dic_train)

        if i % 10 == 0:
            c_train = sess.run(loss, dic_train)
            c_test = sess.run(loss, dic_test)
            # print(i, c)

            preds_train = sess.run(hx, dic_train)
            preds_test = sess.run(hx, dic_test)

            avg_train = show_difference(preds_train, y_train)
            avg_test = show_difference(preds_test, y_test)
            # print('error :', avg)

            c_trains.append(c_train)
            c_tests.append(c_test)
            avg_trains.append(avg_train)
            avg_tests.append(avg_test)

    sess.close()

    show_plot(1, 'LOSS', c_trains, 'train', c_tests, 'test')
    show_plot(2, 'ERROR', avg_trains, 'train', avg_tests, 'test')
    plt.show()


x_train, x_test, y_train, y_test = get_data()
model_forest_fires_regression(x_train, x_test, y_train, y_test)














