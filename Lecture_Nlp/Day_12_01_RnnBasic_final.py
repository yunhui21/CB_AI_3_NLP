# Day_12_01_RnnBasic_final.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 엄청나게 긴 문장에 대해 동작하는 버전을 만드세요


def make_data(long_text, seq_len):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    # 문제
    # long_text를 words 형식으로 바꾸세요
    # words = []
    # for i in range(len(long_text) - seq_len):
    #     words.append(long_text[i:i + seq_len + 1])

    words = [long_text[i:i+seq_len+1] for i in range(len(long_text)-seq_len)]

    x, y = [], []
    for word in words:
        onehot = lb.transform(list(word))

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), lb.classes_


def rnn_basic_final(long_text, seq_len, loop_count):
    x, y, vocab = make_data(long_text, seq_len)

    batch_size, seq_len, n_classes = x.shape
    hidden_size = 21
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # ---- #
    z = tf.layers.dense(outputs, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)
    # ---- #

    optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)                    # (1, 5, 6)
    preds_arg = np.argmax(preds, axis=2)    # (1, 5)

    print(long_text)
    print('*' + ''.join(vocab[preds_arg[0]]), end='')

    for arg in preds_arg[1:]:
        print(''.join(vocab[arg[-1]]), end='')
    print()

    sess.close()


long_text = ("if you want to build a ship,"
             " don't drum up people to collect wood and"
             " don't assign them tasks and work,"
             " but rather teach them to long"
             " for the endless immensity of the sea.")
rnn_basic_final(long_text, seq_len=20, loop_count=300)


