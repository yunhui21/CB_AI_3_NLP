# Day_11_2_RnnBasic_4_different.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 여러 개의 단어에 대해 동작하는 버전을 만드세요.

def make_data(words):
    e1 = preprocessing.LabelBinarizer()
    e1.fit(list(''.join(words)))

    max_len = max([len(w) for  w in words])

    x, y = [], []
    for word in words:
        if len(word) < max_len:
            word += ' '*(max_len - len(word))
        onehot = e1.transform(list(word))

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), e1.classes_



# dense + cross_entropy (앞에 함수 수정한거 없음)
def rnn_basic_4_different(words, loop_count):
    x, y, vocab = make_data(words)

    batch_size, seq_len, n_classes = x.shape
    hidden_size = 21
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

    optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    len_array = [len(w)-1 for w in words]

    for i in range(loop_count):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        preds = sess.run(hx)  # (1, 5, 6)
        preds_arg = np.argmax(preds, axis=2)  # (1, 5)

        # 문제
        # len_array를 사용해서 필요한 만큼 출력하세요.
        # for j in range(len(len_array)):
        #     limited = preds_arg[j]
        #     limited = limited[:len_array[j]]
        #     print(''.join(vocab[limited]), end=' ')
        # print()
        print([''.join(vocab[arg[:cnt]]) for arg, cnt in zip(preds_arg, len_array)])

    sess.close()


rnn_basic_4_different(['tensor','sky','bird'], loop_count=300)

# if you want