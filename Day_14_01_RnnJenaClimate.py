# Day_14_01_RnnJenaClimate.py
import pandas as pd
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()

# 문제
# 파일에 있는 'T(degC)'컬럼을 RNN 알고리즘을 사용하지 말고
# 리니어 리그레션 알고리즘을 평균 오차를 계산하세요.

# baseline model
# print('mae:', np.mean(np.abs(degc[:-144]-degc[144:])))

# 문제
# jena파일에 있는 온데 데이터를  RNN 알고리즘으로 평균 오차를 계산하세요.
# 'p (mbar)', 'T (degC)', 'rho (g/m**3)'



def jena_regression():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values
    degc = degc[:1000]

    x, y = degc[:-144], degc[:-144]

    # float type값으로.
    w = tf.Variable(np.random.normal([1]), dtype=tf.float32)
    b = tf.Variable(np.random.normal([1]), dtype=tf.float32)

    hx = w * x + b

    loss_i = (hx-y)**2 # mean square error
    loss   = tf.reduce_mean(loss_i)

    # optimizer = tf.train.AdamOptimizer(0.1)
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train     = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))

    preds = sess.run(hx)
    # print(preds.shape)#  (856,)-(856,) ->
    print('mae:', np.mean(np.abs(preds-y))) # mae: 2.612549101228096

    sess.close()


def rnn_jena_multi_columns():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    # jena = jena[['p (mbar)','T (degC)','rho (g/m**3)']].values
    jena = [jena['p (mbar)'], jena['rho (g/m**3)'], jena['T (degC)']]
    print(np.array(jena).shape) # (420551, 3)
    jena = np.transpose(jena)

    jena = jena[:1000]
     # (420551, 3)


    seq_len = 144

    rng = range(len(jena) - seq_len)

    x = [jena[s:s + seq_len] for s in rng]
    y = [jena[s + seq_len, -1:] for s in rng]

    x, y = np.float32(x), np.float32(y)
    print(x.shape, y.shape) # (568, 432, 3) (568, 3)

    hidden_size, n_features = 150, 3

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_features])  #
    ph_y = tf.placeholder(tf.float32)  #

    cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)

    hx = tf.layers.dense(outputs[:, -1, :], 1, activation=None)

    loss_i = (hx-ph_y)**2 # mean square error
    loss   = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train     = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 100 # 한번에 처리할 데이터개수
    n_iteration = len(x) // batch_size # epochs한번에 배치사이즈만큼의 횟수

    indices = np.arange(len(x)) # 0 ~ 855 까지 일련번호 숫자.

    for i in range(epochs):
        np.random.shuffle(indices) # 원본데이터는 시계열이지만 슬라이싱으로 작업하면서 데이터가 시계열이 아니게 되었다.
        total = 0
        for j in range(n_iteration): # 60000의 데이터 100으로 나누어
            n1 = j * batch_size         # 0, 100, 200, 300
            n2 = n1 + batch_size        # 100, 200, 300, 400

            xx = x[indices[n1:n2]]
            yy = y[indices[n1:n2]]

            sess.run(train, {ph_x: xx, ph_y: yy})
            total += sess.run(loss,  {ph_x:xx, ph_y:yy})

        print(i, total/n_iteration )

    preds = sess.run(hx, {ph_x: x})
    print(preds.shape, y.shape) #(856, 1) (856, 3)
    print('mae:', np.mean(np.abs(preds-y))) # mae: 2.612549101228096

    # shuffle 적용 mae: 0.74608696
    sess.close()

jena_regression()
# rnn_jena_multi_columns()