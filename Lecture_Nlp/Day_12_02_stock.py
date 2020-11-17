# Day_12_02_stock.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 주식 데이터를 읽어서 1주일 단위로 학습해서 그 다음 날 종가를 예측하세요
# (예측 결과는 그래프로 표시합니다)


def make_data():
    pass


def rnn_stock_1():
    make_data()
    return

    x, y, lb = make_data()
    vocab = lb.classes_

    batch_size, seq_len_old, n_classes = x.shape
    assert(seq_len == seq_len_old)

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_classes])

    hidden_size = 15
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)

    z = tf.layers.dense(outputs, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    preds = sess.run(hx, {ph_x: x})
    preds_arg = np.argmax(preds, axis=2)

    print(long_text)
    print('*' + ''.join(vocab[preds_arg[0]]), end='')

    for arg in preds_arg[1:]:
        print(''.join(vocab[arg])[-1], end='')
    print()

    # -------------------------------- #

    # 문제
    # start_char로부터 시작해서 50개의 미래 문자를 예측하세요
    # [None, seq_len, n_classes]

    # *******p --> ******pk
    # *******k

    current = start_char

    # current가 seq_len보다 작다
    for i in reversed(range(seq_len)):
        xx = lb.transform(list(' ' * i + current))
        # xx = lb.transform(list(' ' * (seq_len - len(current)) + current))
        preds = sess.run(hx, {ph_x: [xx]})
        preds_arg = np.argmax(preds, axis=2)

        idx = preds_arg[0][-1]
        current += vocab[idx]
        print(current, len(current))

    # current가 seq_len보다 크다
    for i in reversed(range(50 - seq_len)):
        xx = lb.transform(list(current[-seq_len:]))
        preds = sess.run(hx, {ph_x: [xx]})
        preds_arg = np.argmax(preds, axis=2)

        idx = preds_arg[0][-1]
        current += vocab[idx]
        print(current, len(current))

    sess.close()


rnn_stock_1()




