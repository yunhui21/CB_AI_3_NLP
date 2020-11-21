# Day_14_01_RnnJenaClimate.py
import pandas as pd
import numpy as np
# import tensorflow as tf           # 1.14 버전
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# 문제
# jena 파일에 있는 "T (degC)" 컬럼을 RNN 알고리즘을 사용하지 말고
# 리니어 리그레션 알고리즘으로 평균 오차를 계산하세요

# baseline model
# print('mae :', np.mean(np.abs(degc[:-144] - degc[144:])))

# 문제
# jean 파일에 있는 온도 데이터를 아래 컬럼을 사용해서 RNN 알고리즘으로 평균 오차를 계산하세요
# "p (mbar)", "T (degC)", "rho (g/m**3)"


def jena_regression():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values
    degc = degc[:1000]

    x, y = degc[:-144], degc[144:]

    w = tf.Variable(np.random.normal([1]), dtype=tf.float32)
    b = tf.Variable(np.random.normal([1]), dtype=tf.float32)

    hx = w * x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))

    print('-' * 30)

    preds = sess.run(hx)
    print('mae :', np.mean(np.abs(preds - y)))  # (856,) - (856,)
    sess.close()


def rnn_jena_multi_columns():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    jena = [jena["p (mbar)"], jena["rho (g/m**3)"], jena["T (degC)"]]
    jena = np.transpose(jena)

    # jena = jena[["p (mbar)", "rho (g/m**3)", "T (degC)"]].values

    print(np.array(jena).shape, jena.dtype)     # (420551, 3) float64

    jena = jena[:1000]

    seq_len = 144

    # ---------------------- #
    # 주식에서 가져온 코드
    rng = range(len(jena) - seq_len)

    x = [jena[s:s+seq_len] for s in rng]        # 3차원
    y = [jena[s+seq_len, -1:] for s in rng]     # 2차원

    x, y = np.float32(x), np.float32(y)
    print(x.shape, y.shape)                     # (856, 144, 3) (856, 1)

    # ---------------------- #
    # mnist에서 가져온 코드
    hidden_size, n_features = 150, 3

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_features])
    ph_y = tf.placeholder(tf.float32)

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    hx = tf.layers.dense(outputs[:, -1, :], 1, activation=None)

    loss_i = (hx - ph_y) ** 2       # mse: mean square error
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 10
    batch_size = 100
    n_iteration = len(x) // batch_size

    indices = np.arange(len(x))

    for i in range(epochs):
        np.random.shuffle(indices)
        total = 0
        for j in range(n_iteration):
            n1 = j * batch_size         #   0 100 200 300
            n2 = n1 + batch_size        # 100 200 300 400

            xx = x[indices[n1:n2]]
            yy = y[indices[n1:n2]]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss, {ph_x: xx, ph_y: yy})

        print(i, total / n_iteration)

    print('-' * 30)

    preds = sess.run(hx, {ph_x: x})
    print(preds.shape, y.shape)
    print('mae :', np.mean(np.abs(preds - y)))
    sess.close()


jena_regression()
# rnn_jena_multi_columns()
