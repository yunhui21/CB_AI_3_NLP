# Day_19_03_LogisticRegression.py
import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt


def logistic_regression():
    #       공부   출석
    x = [[1., 1., 2.],  # 탈락
         [1., 2., 1.],
         [1., 4., 5.],  # 통과
         [1., 5., 4.],
         [1., 8., 9.],
         [1., 9., 8.]]
    y = [[0], [0], [1], [1], [1], [1]]
    # y = np.int32(y)
    y = np.float32(y)

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (6, 1) = (6, 3) @ (3, 1)
    z = tf.matmul(x, w)
    hx = tf.sigmoid(z)  # 1 / (1 + tf.exp(-z))

    # loss_i = y * -tf.log(hx) + (1 - y) * -tf.log(1 - hx)
    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    print(preds)

    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = y.reshape(-1)

    print(preds_1)
    print(y_1)

    print('acc :', np.mean(preds_1 == y_1))
    sess.close()


# 문제
# iris 데이터에서 2개의 품종을 골라서
# 70%로 학습하고 30%에 대해 예측한 결과를 정확도로 표시하세요
def logistic_regression_iris():
    x, y = datasets.load_iris(return_X_y=True)
    y = y.reshape(-1, 1)

    x = x[:100]
    y = y[:100]
    # x = x[50:]
    # y = y[50:]
    # y[y == 2] = 0

    # print(y)

    x = np.float32(x)
    y = np.float32(y)

    # print(type(x), type(y))
    # print(x.shape, y.shape)         # (100, 4) (100, 1)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    w = tf.Variable(tf.random_uniform([x.shape[1], 1]))     # 4
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    # (100, 1) = (100, 4) @ (4, 1)
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = y_test.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    print(preds_1)
    print(y_1)

    print('acc :', np.mean(preds_1 == y_1))
    sess.close()

    # plt.plot(x[:, 0], x[:, 2], 'ro')
    plt.scatter(x[:, 0], x[:, 2], c=y)
    plt.show()


# logistic_regression()
logistic_regression_iris()
