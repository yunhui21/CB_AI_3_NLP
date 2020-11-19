# Day_11_2_RnnBasic_4_different.py

import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 여러 개의 단어에 대해 동작하는 버전을 만드세요.

def make_data_1(long_text, seq_len):
    e1 = preprocessing.LabelBinarizer()
    onehot = e1.fit_transform(list(long_text))
    print(onehot.shape) # (171, 25)

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

    # 1번
    # rng = [(i, i+seq_len) for i in range(len(onehot)- seq_len)]
    # print(rng[-1]) # boundery check 171 (151, 171)
    # print(long_text[rng[-1][0]:rng[-1][1]]) # immensity of the sea
    #
    # xx = [x[s:e] for s, e in rng]
    # yy = [y[s:e] for s, e in rng]

    # 2번
    # rng = [i for i in range(len(onehot)-seq_len)]
    rng = range(len(onehot)-seq_len)

    xx = [x[s:s+seq_len] for s in rng]
    yy = [y[s:s+seq_len] for s in rng]

    # 3번
    # xx = [x[s:s+seq_len] for s in range(len(onehot)- seq_len)]
    # yy = [y[s:s+seq_len] for s in range(len(onehot)- seq_len)]

    # for i in xx:
    #    print(i.shape) # (20, 25) (19, 25)
    # exit(-1)
    return np.float32(xx), np.int32(yy), e1.classes_


def make_data_2(long_text, seq_len):
    e1 = preprocessing.LabelBinarizer()
    e1.fit(list(long_text))

    # 문제
    # long_text를 words 형식으로 바꾸세요.

    # words = []
    # for i in range(len(long_text)-seq_len):
    #    words.append(long_text[i:i + seq_len+1])

    words =[long_text[i:i + seq_len+1] for i in range(len(long_text)-seq_len)]

    x, y = [], []
    for word in words: # list형태로 들어오기만 하면 된다.
       onehot = e1.transform(list(word)) # abcd, efgh

       x.append(np.float32(onehot[:-1]))
       y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), e1.classes_

# dense + cross_entropy (앞에 함수 수정한거 없음)
def rnn_basic_5(long_text,seq_len, loop_count):
    x, y, vocab = make_data_2(long_text, seq_len)

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

    for i in range(loop_count):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)  # (1, 5, 6)
    preds_arg = np.argmax(preds, axis=2)  # (1, 5)


    # for arg in preds_arg:
        # print(''.join(vocab[arg]))
    print(long_text)
    print('*'+''.join(vocab[preds_arg[0]]), end='')

    for arg in preds_arg[1:]:
        print(''.join(vocab[arg[-1]]), end='')
    print()


    sess.close()

long_text = ("if you want to build a ship,"
             " don't drum up people to collect wood and"
             " don't assign them tasks and work,"
             " but rather teach them to long"
             " for the endless immensity of the sea.")
rnn_basic_5(long_text, seq_len=20, loop_count=300)
