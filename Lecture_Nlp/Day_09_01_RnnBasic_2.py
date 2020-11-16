# Day_09_01_RnnBasic_2.py
import tensorflow as tf
import numpy as np

np.set_printoptions(linewidth=1000)


# 문제
# dense 레이어를 rnn 레이어로 변환하세요
# (소프트맥스 리그레션 모델을 RNN 모델로 변환하세요)
def rnn_basic_2_1():
    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
        [0, 0, 0, 1, 0, 0],  # 3, r
    ]
    x = np.float32(x)

    # 문제
    # x를 3차원으로 만드세요 (1, 5, 6)
    # x = x.reshape(1, 5, 6)
    # x = x.reshape(1, x.shape[0], x.shape[1])
    # x = x.reshape(1, *x.shape)
    # x = x.reshape((1,) + x.shape)
    x = x[np.newaxis]

    hidden_size = 6
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    print(outputs.shape)    # (1, 5, 7)

    # w = tf.Variable(tf.random.uniform([6, 6]))
    # b = tf.Variable(tf.random.uniform([6]))
    #
    # # (5, 6) = (5, 6) @ (6, 6)
    # z = tf.matmul(x, w) + b

    z = outputs[0]
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    word = np.array(list('enorst'))

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        print(''.join(word[preds_arg]))

    sess.close()


# 문제
# rnn 레이어 다음에 앞에서 사용했던 dense 레이어를 연결하세요
def rnn_basic_2_2():
    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
        [0, 0, 0, 1, 0, 0],  # 3, r
    ]
    x = np.float32(x)

    # 문제
    # x를 3차원으로 만드세요 (1, 5, 6)
    # x = x.reshape(1, 5, 6)
    # x = x.reshape(1, x.shape[0], x.shape[1])
    # x = x.reshape(1, *x.shape)
    # x = x.reshape((1,) + x.shape)
    x = x[np.newaxis]

    hidden_size = 11
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    print(outputs.shape)    # (1, 5, 31)

    w = tf.Variable(tf.random.uniform([hidden_size, 6]))
    b = tf.Variable(tf.random.uniform([6]))

    # (5, 6) = (5, 31) @ (31, 6)
    z = tf.matmul(outputs[0], w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    word = np.array(list('enorst'))

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        print(''.join(word[preds_arg]))

    sess.close()


# 문제
# y 데이터를 단순 인코딩으로 바꿔주세요
def rnn_basic_2_3():
    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [0, 1, 4, 2, 3]
    # y = [
    #     [1, 0, 0, 0, 0, 0],  # 0, e
    #     [0, 1, 0, 0, 0, 0],  # 1, n
    #     [0, 0, 0, 0, 1, 0],  # 4, s
    #     [0, 0, 1, 0, 0, 0],  # 2, o
    #     [0, 0, 0, 1, 0, 0],  # 3, r
    # ]
    x = np.float32(x)
    x = x[np.newaxis]

    hidden_size = 11
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    w = tf.Variable(tf.random.uniform([hidden_size, 6]))
    b = tf.Variable(tf.random.uniform([6]))

    z = tf.matmul(outputs[0], w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    word = np.array(list('enorst'))

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        print(''.join(word[preds_arg]))
        # print('acc :', np.mean(preds_arg == y))

    sess.close()


# 문제
# rnn 레이어 2개, dense 레이어 2개로 모델을 만드세요
def rnn_basic_2_4():
    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [0, 1, 4, 2, 3]

    x = np.float32(x)
    x = x[np.newaxis]

    hidden_size_1 = 11
    cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size_1, name='c1')
    outputs_1, _states_1 = tf.nn.dynamic_rnn(cell_1, x, dtype=tf.float32)

    hidden_size_2 = 11
    cell_2 = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size_2, name='c2')
    outputs_2, _states_2 = tf.nn.dynamic_rnn(cell_2, outputs_1, dtype=tf.float32)

    # ------------------------------ #

    w1 = tf.Variable(tf.random.uniform([hidden_size_2, 6]))
    b1 = tf.Variable(tf.random.uniform([6]))

    # (5, 6) = (5, 11) @ (11, 6)
    z1 = tf.matmul(outputs_2[0], w1) + b1

    w2 = tf.Variable(tf.random.uniform([6, 6]))
    b2 = tf.Variable(tf.random.uniform([6]))

    # (5, 6) = (5, 6) @ (6, 6)
    z = tf.matmul(z1, w2) + b2
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    word = np.array(list('enorst'))

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        print(''.join(word[preds_arg]))

    sess.close()


def rnn_basic_2_5():
    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [0, 1, 4, 2, 3]

    x = np.float32(x)
    x = x[np.newaxis]

    hidden_size_1 = 7
    cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size_1, name='c1')
    outputs_1, _states_1 = tf.nn.dynamic_rnn(cell_1, x, dtype=tf.float32)

    hidden_size_2 = 11
    cell_2 = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size_2, name='c2')
    outputs_2, _states_2 = tf.nn.dynamic_rnn(cell_2, outputs_1, dtype=tf.float32)

    print(outputs_1.shape)      # (1, 5, 7)
    print(outputs_2.shape)      # (1, 5, 11)

    # ------------------------------ #

    w1 = tf.Variable(tf.random.uniform([hidden_size_2, 17]))
    b1 = tf.Variable(tf.random.uniform([17]))

    # (5, 17) = (5, 11) @ (11, 17)
    z1 = tf.matmul(outputs_2[0], w1) + b1

    w2 = tf.Variable(tf.random.uniform([17, 6]))
    b2 = tf.Variable(tf.random.uniform([6]))

    # (5, 6) = (5, 17) @ (17, 6)
    z = tf.matmul(z1, w2) + b2
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    word = np.array(list('enorst'))

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # -------------------- #

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        print(''.join(word[preds_arg]))

    sess.close()


# rnn_basic_2_1()
# rnn_basic_2_2()
# rnn_basic_2_3()
# rnn_basic_2_4()
rnn_basic_2_5()
