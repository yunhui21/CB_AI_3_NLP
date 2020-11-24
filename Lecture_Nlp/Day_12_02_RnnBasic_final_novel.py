# Day_12_02_RnnBasic_final_novel.py
import nltk
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 구텐베르그에 있는 소설 엠마를 가져와서 단어로 토큰화하세요

# 문제
# 토큰화된 소설을 사용해서 여러분의 소설을 써보세요 (char rnn -> word rnn)


def make_data(tokens, seq_len):
    lb = preprocessing.LabelBinarizer()
    lb.fit(tokens)

    words_list = [tokens[i:i+seq_len+1] for i in range(len(tokens)-seq_len)]

    x, y = [], []
    for words in words_list:
        onehot = lb.transform(words)

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), lb


def rnn_basic_final_novel(tokens, seq_len, loop_count, start_token):
    x, y, lb = make_data(tokens, seq_len)
    vocab = lb.classes_

    batch_size, seq_len, n_classes = x.shape
    hidden_size = 21

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_classes])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)

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
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print('-' * 30)

    # -------------------------------------------- #

    current = [' '] * (seq_len - 1) + [start_token]

    for i in range(50):
        xx = lb.transform(current[-seq_len:])
        xx = [xx]

        preds = sess.run(hx, {ph_x: xx})
        preds_arg = np.argmax(preds, axis=2)

        idx = preds_arg[0][-1]
        current.append(vocab[idx])

        print(' '.join(current[seq_len-1:]))

    sess.close()


# print(nltk.corpus.gutenberg.fileids())

emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
emma = emma.lower()

tokens = nltk.regexp_tokenize(emma, r'\w+')
print(tokens[:10])

rnn_basic_final_novel(tokens[:100], seq_len=20, loop_count=300, start_token=tokens[37])

# a = [1, 2, 3]
# a += 'abc'
# print(a)
#
# a.append('abc')
# print(a)
