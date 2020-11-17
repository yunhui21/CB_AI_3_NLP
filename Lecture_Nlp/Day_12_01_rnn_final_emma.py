# Day_12_01_rnn_final_emma.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 이전 코드를 구텐베르그의 엠마 소설을 이용하도록 수정하세요


def make_data(long_text, seq_len):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    # 문제
    # 긴 문자열을 여러 개의 단어로 구성된 리스트로 만드세요
    words =[long_text[i:i+seq_len+1]
            for i in range(len(long_text) - seq_len)]

    x, y = [], []
    for word in words:
        onehot = lb.transform(list(word))
        # print(onehot, end='\n\n')

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), lb


def rnn_basic_final(long_text, seq_len, loop_count, start_char):
    x, y, lb = make_data(long_text, seq_len)
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

    current = ' ' * (seq_len-1) + start_char

    # current가 seq_len보다 작다
    for i in reversed(range(50)):
        xx = lb.transform(list(current[-seq_len:]))
        # xx = lb.transform(list(' ' * (seq_len - len(current)) + current))
        preds = sess.run(hx, {ph_x: [xx]})
        preds_arg = np.argmax(preds, axis=2)

        idx = preds_arg[0][-1]
        current += vocab[idx]
        print(current[seq_len-1:])

    # # current가 seq_len보다 크다
    # for i in reversed(range(50 - seq_len)):
    #     xx = lb.transform(list(current[-seq_len:]))
    #     preds = sess.run(hx, {ph_x: [xx]})
    #     preds_arg = np.argmax(preds, axis=2)
    #
    #     idx = preds_arg[0][-1]
    #     current += vocab[idx]
    #     print(current, len(current))

    sess.close()


long_text = ("if you want to build a ship,"
             " don't drum up people to collect wood and"
             " don't assign them tasks and work,"
             " but rather teach them to long"
             " for the endless immensity of the sea.")
rnn_basic_final(long_text, seq_len=20, loop_count=300, start_char='t')
