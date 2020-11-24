# Day_13_01_RnnMnist.py
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf


def get_mnist():
    mnist = input_data.read_data_sets('mnist', one_hot=False)

    print(mnist.train.images.shape)         # (55000, 784)
    print(mnist.validation.images.shape)    # (5000, 784)
    print(mnist.test.images.shape)          # (10000, 784)

    print(mnist.train.labels.shape)         # (55000,)
    print(mnist.validation.labels.shape)    # (5000,)
    print(mnist.test.labels.shape)          # (10000,)

    # 문제
    # train과 validation 데이터를 묶어서 train으로 만드세요
    # (60000, 784)  (60000,)

    # x_train = np.vstack([mnist.train.images, mnist.validation.images])
    # y_train = np.hstack([mnist.train.labels, mnist.validation.labels])

    # x_train = np.concatenate([mnist.train.images, mnist.validation.images], axis=0)
    x_train = np.concatenate([mnist.train.images, mnist.validation.images])
    y_train = np.concatenate([mnist.train.labels, mnist.validation.labels])

    print(x_train.shape, y_train.shape)     # (60000, 784) (60000,)

    x_test = mnist.test.images
    y_test = mnist.test.labels

    print(y_train.dtype)        # uint8

    y_train = np.int32(y_train)
    y_test = np.int32(y_test)

    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    return x_train, x_test, y_train, y_test
    # return x_train, y_train, x_test, y_test


# ----------------------------------------- #

# 문제
# mnist 데이터셋을 RNN 알고리즘을 사용해서 test 데이터에 대한 정확도를 계산하세요
def rnn_mnist_basic():
    x_train, x_test, y_train, y_test = get_mnist()

    seq_len, n_classes, n_features = 28, 10, 28
    hidden_size = 21

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_features])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)

    z = tf.layers.dense(outputs[:, -1, :], n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y_train)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    print('-' * 30)

    preds = sess.run(hx, {ph_x: x_test})
    print(preds.shape)

    preds_arg = np.argmax(preds, axis=1)

    print('acc :', np.mean(preds_arg == y_test))
    sess.close()


# 문제
# 앞에서 만든 코드를 미니배치 방식으로 수정하세요 (멀티 RNN 적용)
def rnn_mnist_minibatch():
    x_train, x_test, y_train, y_test = get_mnist()

    seq_len, n_classes, n_features = 28, 10, 28
    hidden_size = 150

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_features])
    ph_y = tf.placeholder(tf.int32)

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    # cells = [
    #     tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size),
    #     tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size),
    # ]

    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    z = tf.layers.dense(outputs[:, -1, :], n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 10
    batch_size = 100
    n_iteration = len(x_train) // batch_size

    for i in range(epochs):
        total = 0
        for j in range(n_iteration):
            n1 = j * batch_size         #   0 100 200 300
            n2 = n1 + batch_size        # 100 200 300 400

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, total / n_iteration)

    print('-' * 30)

    preds = sess.run(hx, {ph_x: x_test})
    print(preds.shape)

    preds_arg = np.argmax(preds, axis=1)

    print('acc :', np.mean(preds_arg == y_test))    # acc : 0.9727
    sess.close()


# rnn_mnist_basic()
# rnn_mnist_minibatch()
