# Day_10_01_RnnBasic_3.py
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

def make_data_1():
    vocab = np.array(list('enorst'))

    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [0, 1, 4, 2, 3]
    y = np.reshape(y, [1,5])
    x = np.float32(x)
    x = x[np.newaxis]

    return x, tf.constant(y), vocab

# matmul + sequence_loss
def rnn_basic_3_1():
    x, y, vocab = make_data_1()

    batch_size, seq_len, n_classes = x.shape
    hidden_size = 11
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    w = tf.Variable(tf.random.uniform([batch_size, hidden_size, n_classes]))
    b = tf.Variable(tf.random.uniform([n_classes]))

    # 문제
    # 아래 두 줄을 matmul 함수 한 개로 대체하세요.
    # z = tf.matmul(outputs[0], w) + b
    # z = tf.reshape(z, (1, z.shape[0], z.shape[1]) )

    z = tf.matmul(outputs, w) + b
    hx = tf.nn.softmax(z)

    # loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    # loss = tf.reduce_mean(loss_i)
    w = tf.ones([1, y.shape[1]]) # 1차원
    loss = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=w)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())


    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)                    # (1, 5, 6)
        preds_arg = np.argmax(preds, axis=2)    # (1, 5)

        print(''.join(vocab[preds_arg[0]]))

    sess.close()



# -------------------------------- #

# 문제
# 입력된 단어를 x, y, vocab으로 변환하는 함수를 만드세요.
# 'tensor' ->
# [5, 0, 1, 4, 2, 3] ->
# [5, 0, 1, 4, 2], [0, 1, 4, 2, 3] ->
# x를 원핫 벡터로 변환
# 파이썬(넘파이)버전

def make_data_2(word):
    vocab = np.array(sorted(set(word)))
    print(vocab)

    chr2idx = {c: i for i,c in enumerate(vocab)}
    idx2chr = {i: c for i,c in enumerate(vocab)} # 없어도 되는 변수.vocab이 있으니까.
    # {'e': 0, 'n': 1, 'o': 2, 'r': 3, 's': 4, 't': 5}
    # {0: 'e', 1: 'n', 2: 'o', 3: 'r', 4: 's', 5: 't'}
    print(chr2idx)
    print(idx2chr)

    indices = [chr2idx[c] for c in word] # word 문자열
    # [5, 0, 1, 4, 2, 3]
    print(indices)

    # 좋아보이지만, 토큰이 많아지면 느려지는 코드
    # vocab = np.array(sorted(set(word)))
    # indices = [vocab.index(c) for c in word]
    # print(indices)

    x = indices[:-1]
    y = indices[1:]
    print(x, y)         # [5, 0, 1, 4, 2] [0, 1, 4, 2, 3]

    # onehot = np.zeros([len(vocab), len(vocab)], dtype=np.int32)
    # onehot[range(len(vocab)), range(len(vocab))] = 1

    onehot = np.eye(len(vocab), dtype=np.int32)
    print(onehot)
    print()

    # x = onehot[[1, 4, 0]]
    x = onehot[x]
    # print(x)
    #
    # exit(-1)

    y = np.reshape(y, [1, len(y)])
    x = np.float32(x)
    x = x[np.newaxis]

    return x, y, vocab

# dense + cross_entropyw
def rnn_basic_3_2(word):
    x, y, vocab = make_data_2(word)

    batch_size, seq_len, n_classes = x.shape
    hidden_size = 11
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # ---------------------- #
    z  = tf.layers.dense(outputs, n_classes, activation=None)
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


    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)                    # (1, 5, 6)
        preds_arg = np.argmax(preds, axis=2)    # (1, 5)

        print(''.join(vocab[preds_arg[0]]))

    sess.close()


# -------------------------------- #
# 사이킷런 버전
def make_data_3(word):
    e1 = preprocessing.LabelBinarizer()
    onehot = e1.fit_transform(list(word))

    x = np.float32(onehot[:-1])
    x = x[np.newaxis]

    y = np.argmax(onehot[1:], axis=1)
    y = y[np.newaxis]
    print(e1.classes_)

    return x, y, e1.classes_


# dense + cross_entropy (앞에 함수 수정한거 없음)
def rnn_basic_3_3(word,loop_count):
    x, y, vocab = make_data_3(word)

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

        print(''.join(vocab[preds_arg[0]]))

    sess.close()

# rnn_basic_3_1()

# rnn_basic_3_2('tensor')
# rnn_basic_3_2('hello')

# rnn_basic_3_3('deep learning', loop_count=100)