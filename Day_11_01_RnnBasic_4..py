import tensorflow as tf
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# 여러 개의 단어에 대해 동작하는 버전을 만드세요.

def make_data(words):
        e1 = preprocessing.LabelBinarizer()
        e1.fit(list(''.join(words)))

        x, y = [], []
        for word in words:
            onehot = e1.transform(list(word)) # abcd, efgh

            x.append(np.float32(onehot[:-1]))
            y.append(np.argmax(onehot[1:], axis=1))
        # print(e1.classes_)
        # exit(-1) # ['c' 'e' 'f' 'l' 'n' 'o' 'r' 's' 't' 'w' 'y']

        return np.float32(x), np.int32(y), e1.classes_


# dense + cross_entropy (앞에 함수 수정한거 없음)
def rnn_basic_4(words,loop_count):
    x, y, vocab = make_data(words)

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
        # print(preds_arg.shape) # (3, 5)

        # 문제
        # 예측 결과가 모두 표시되도록 수정하세요.

        # for i in range(len(preds_arg):
        #     print(''.join(vocab[preds_arg[j]]), end = ' ')
        # print()

        # for arg in (preds_arg):
        #     print(''.join(vocab[arg]), end = ' ')
        # print()

        print([''.join(vocab[arg]) for arg in preds_arg])
    sess.close()

rnn_basic_4(['tensor','yellow','coffee'], loop_count=100)
