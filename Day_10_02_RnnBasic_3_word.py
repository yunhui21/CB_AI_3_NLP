# Day_10_02_RnnBasic_3_word.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
# 문제
# 아래 문장에 대해 앞에서 배웠던 chr rnn 알고리즘을 적용하세요.
# grove -> g: r, r: o
# "오늘도 그로브에 가서 공부했다."
np.set_printoptions(linewidth=1000)


def make_data(text):

    e1 = preprocessing.LabelBinarizer()
    onehot = e1.fit_transform(text.split())

    # print(e1.classes_) # ['가서' '공배했다.' '그로브에' '오늘도']
    # exit(-1)

    x = np.float32(onehot[:-1])
    x = x[np.newaxis]

    y = np.argmax(onehot[1:], axis=1)
    y = y[np.newaxis]
    print(e1.classes_)

    return x, y, e1.classes_


# dense + cross_entropy (앞에 함수 수정한거 없음)
def rnn_word(text,loop_count):
    x, y, vocab = make_data(text)

    batch_size, seq_len, n_classes = x.shape
    hidden_size = 11
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # ---------------------- #
    z = tf.layers.dense(outputs, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)
    # ---------------------- #
    # w = tf.ones([1, y.shape[1]]) # 1차원
    # loss = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=w)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        preds = sess.run(hx)  # (1, 5, 6)
        preds_arg = np.argmax(preds, axis=2)  # (1, 5)

        print(' '.join(vocab[preds_arg[0]]))

    sess.close()
text = '오늘도 그로브에 가서 공배했다.'
rnn_word(text, loop_count=100)
