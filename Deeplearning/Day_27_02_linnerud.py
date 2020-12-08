# Day_27_02_linnerud.py
import numpy as np
from sklearn import model_selection, preprocessing, datasets
import tensorflow as tf


# 문제 1
# linnerud 데이터에 대해서 결과를 예측하세요 (train, test 구분하지 않습니다)

# 문제 2
# 개별적으롬 모델을 구성하지 말고 1개의 모델로 결과를 예측하세요


def basic():
    rud = datasets.load_linnerud()
    print(rud.keys())
    # ['data', 'feature_names', 'target', 'target_names', 'DESCR', 'data_filename', 'target_filename']

    print(rud['feature_names'])     # ['Chins', 'Situps', 'Jumps']
    print(rud['target_names'])      # ['Weight', 'Waist', 'Pulse']

    print(rud['data'])
    # [[  5. 162.  60.]
    #  [  2. 110.  60.]
    #  [ 12. 101. 101.]

    print(rud['target'])
    # [[191.  36.  50.]
    #  [189.  37.  52.]
    #  [193.  38.  58.]

    print(rud['data'].shape)        # (20, 3)


def show_difference(preds, labels):
    preds_1 = preds.reshape(-1)
    ytest_1 = labels.reshape(-1)

    diff = preds_1 - ytest_1
    diff_abs = np.abs(diff)

    avg = np.mean(diff_abs)
    print('오차 평균 :', avg)


def show_difference_all(preds, labels):
    diff = preds - labels
    diff_abs = np.abs(diff)

    avg = np.mean(diff_abs, axis=0)
    print('오차 평균 :', avg)


def model_linnerud_regression(x, y):
    n_features = x.shape[1]
    n_classes = 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (20, 1) = (20, 3) @ (3, 1)
    hx = tf.matmul(x, w) + b

    # (20, 1) = (20, 1) - (20, 1)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        # print(i, sess.run(loss))

    preds = sess.run(hx)
    show_difference(preds, y)
    sess.close()


def model_linnerud_regression_all_1(x, y):
    n_features = x.shape[1]
    n_classes = 3
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (20, 3) = (20, 3) @ (3, 3)
    hx = tf.matmul(x, w) + b

    # (20, 3) = (20, 3) - (20, 3)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=0)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    show_difference_all(preds, y)
    sess.close()


def model_linnerud_regression_all_2(x, y):
    n_features = x.shape[1]
    n_classes = 1
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (20, 1) = (20, 3) @ (3, 1)
    hx = tf.matmul(x, w) + b

    # (20, 3) = (20, 1) - (20, 3)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i, axis=0)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    show_difference_all(preds, y)
    sess.close()


# basic()

x, y = datasets.load_linnerud(return_X_y=True)

x = np.float32(x)

# model_linnerud_regression(x, y[:, 0].reshape(-1, 1))

# model_linnerud_regression(x, y[:, 0:1])
# model_linnerud_regression(x, y[:, 1:2])
# model_linnerud_regression(x, y[:, 2:3])
#
# for i in range(3):
#     model_linnerud_regression(x, y[:, i:i+1])

model_linnerud_regression_all_1(x, y)
# 오차 평균 : [98.78456612 40.75342855 19.48329525]

# model_linnerud_regression_all_2(x, y)
# 오차 평균 : [101.71729527  43.08439007  25.81914387]





