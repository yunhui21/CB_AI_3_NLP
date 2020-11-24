# Day_13_02_RnnJenaClimate.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf           # 1.14 버전
import tensorflow.compat.v1 as tf   # 2.x 버전
tf.disable_eager_execution()

# 문제
# jena 파일에 있는 "T (degC)" 컬럼을 그래프로 그려주세요

# 문제
# jean 파일에 있는 온도 데이터를 RNN 알고리즘으로 평균 오차를 계산하세요


def show_jena():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    print(jena)

    # degc = jena['T (degC)']
    # degc.plot()

    degc = jena['T (degC)'].values
    plt.plot(range(len(degc)), degc)
    print(type(degc))

    # plt.show()

    # baseline model
    print('mae :', np.mean(np.abs(degc[:-144] - degc[144:])))   # mae : 2.612549101228096


def rnn_jena():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values
    degc = degc[:1000]

    seq_len = 144

    # ---------------------- #
    # 주식에서 가져온 코드
    rng = range(len(degc) - seq_len)

    x = [degc[s:s+seq_len] for s in rng]    # 2차원
    y = [degc[s+seq_len] for s in rng]      # 1차원

    x, y = np.float32(x), np.float32(y)
    print(x.shape, y.shape)                 # (856, 144) (856,)

    x = x[:, :, np.newaxis]
    y = y.reshape(-1, 1)
    print(x.shape, y.shape)                 # (856, 144, 1) (856, 1)

    # ---------------------- #
    # mnist에서 가져온 코드
    hidden_size = 150

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, 1])
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

    indices = np.arange(len(x))         # 0 ~ 855

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
    print('mae :', np.mean(np.abs(preds - y)))
    # mae : 1.2364682  (셔플 안함)
    # mae : 0.45930803 (셔플 적용)
    sess.close()


# show_jena()
rnn_jena()

# [0, 1, 2, 3, 4]
# [1, 2, 3]
# [2, 3, 4]
# [0, 1, 2]
