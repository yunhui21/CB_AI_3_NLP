# Day_09_01_RnnBasic_2.py
import tensorflow as tf
import numpy as np

np.set_printoptions(linewidth=1000)


# 문제
# dense 레이어를 rnn 레이어로 변환하세요.
#

def rnn_basic_2_1():
    # tensor(t: 1 0 0 0 0 0, e: 0 1 0 0 0 0)    # bad
    # enorst(t: 0 0 0 0 0 1, e: 1 0 0 0 0 0)    # good
    # x: tenso
    # y: ensor
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
    # x를 3차원으로 만드세요(1, 5, 6)
    # x = x.reshape(1, 5, 6)
    # x = x.reshape(1, x.shape[0], x.shape[1])
    # x = x.reshape(1, *x.shape)
    # x = x.reshape(1,) + x.shape
    x = x[np.newaxis]

    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    output, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(output.shape) # (1, 5, 7) hidden_size값이 들어간다.

    # w = tf.Variable(tf.random.uniform([6, 6]))
    # b = tf.Variable(tf.random.uniform([6]))
    #
    # # (5, 6) = (5, 6) @ (6, 6)
    # z = tf.matmul(x, w) + b

    hx = tf.nn.softmax(output[0]) # z -> output[0] 3차원으로 변환

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(logits=output[0], labels=y)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    word = np.array(list('enorst'))

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss), end=' ')

        # ---------------------- #

        preds = sess.run(hx)
        preds_arg = np.argmax(preds, axis=1)

        print(''.join(word[preds_arg]))
    sess.close()

rnn_basic_2_1()