# Day_11_01_rnn_4.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 여러 개의 단어에 대해 동작하는 버전을 만드세요


# 사이킷런 버전
def make_data(words):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(''.join(words)))
    # print(lb.classes_)

    x, y = [], []
    for word in words:
        onehot = lb.transform(list(word))
        # print(onehot, end='\n\n')

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), lb.classes_


# 여러 개의 단어에 대해 동작하는 버전
def rnn_basic_4(words, loop_count):
    x, y, vocab = make_data(words)

    batch_size, seq_len, n_classes = x.shape    # (1, 5, 6)
    hidden_size = 5
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    z = tf.layers.dense(outputs, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=2)
        # print(preds_arg.shape)        # (3, 5)

        # 문제
        # 예측 결과를 정확하게 표시하세요
        # print([''.join(vocab[preds_arg[j]]) for j in range(len(preds_arg))])
        print([''.join(vocab[arg]) for arg in preds_arg])

    sess.close()


rnn_basic_4(['tensor', 'coffee', 'forest'], loop_count=300)
