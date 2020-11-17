# Day_11_03_rnn_5.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 긴 문자열에 대해 동작하는 버전을 만드세요


def make_data_1(long_text, seq_len):
    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list(long_text))

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

    # 1번
    # rng = [(i, i+seq_len) for i in range(len(x) - seq_len + 1)]
    # print(len(long_text), rng[-1])
    # print(long_text[rng[-1][0]:rng[-1][1]])
    #
    # xx = [x[s:e] for s, e in rng]
    # yy = [y[s:e] for s, e in rng]

    # 2번
    # rng = [i for i in range(len(x) - seq_len + 1)]
    #
    # xx = [x[s:s+seq_len] for s in rng]
    # yy = [y[s:s+seq_len] for s in rng]

    # 3번
    limit = len(x) - seq_len + 1
    xx = [x[s:s+seq_len] for s in range(limit)]
    yy = [y[s:s+seq_len] for s in range(limit)]

    return np.float32(xx), np.int32(yy), lb.classes_


def make_data_2(long_text, seq_len):
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

    return np.float32(x), np.int32(y), lb.classes_


def rnn_basic_5(long_text, seq_len, loop_count):
    # x, y, vocab = make_data_1(long_text, seq_len)
    x, y, vocab = make_data_2(long_text, seq_len)

    batch_size, seq_len_old, n_classes = x.shape    # (1, 5, 6)
    assert(seq_len == seq_len_old)

    hidden_size = 15
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    z = tf.layers.dense(outputs, n_classes, activation=None)
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    preds_arg = np.argmax(preds, axis=2)

    # 문제
    # 예측 결과를 정확하게 표시하세요
    print(long_text)
    print('*' + ''.join(vocab[preds_arg[0]]), end='')

    for arg in preds_arg[1:]:
        print(''.join(vocab[arg])[-1], end='')
    print()

    # for arg in preds_arg:
    #     print(''.join(vocab[arg]))
    sess.close()


long_text = ("if you want to build a ship,"
             " don't drum up people to collect wood and"
             " don't assign them tasks and work,"
             " but rather teach them to long"
             " for the endless immensity of the sea.")
rnn_basic_5(long_text, seq_len=20, loop_count=300)

# 'if you want'
# 'if y'
# 'you '
# ' wan'

# 'if you want'
# 'if y'
# 'f yo'
# ' you'

# 캐글 초대 링크
# https://www.kaggle.com/t/81398e1d4a3e4d85a0778ccecb920364

