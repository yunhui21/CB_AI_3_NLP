# Day_13_02_RnnJenaClimate.py
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 문제
# jena
# 파일에 있는 'T(decG)'컬럼을 그래프로 그려주세요.

# 문제
# jena 파일에 있는 온도 데이터를 RNN 알고리즘으로 평균 오차를 계산하세요.

def show_Jena():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values

    plt.plot(range(len(degc)), degc)
    # plt.show()
    print(type(degc)) # <class 'numpy.ndarray'>

    # baseline model
    print('mae:', np.mean(np.abs(degc[:-144]-degc[144:]))) # mae: 2.612549101228096


def rnn_jena():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values
    degc = degc[:1000]

    seq_len = 144 * 3
    # stock에서 가져온 코드
#   #------------------------------------#
    rng = range(len(degc) - seq_len)  # +1 삭제(8번째가 y라서)

    x = [degc[s:s+seq_len] for s in rng]  # 2차원
    y = [degc[s+seq_len] for s in rng]    # 1차원 column 1ro

    x, y = np.float32(x), np.float32(y)
    print(x.shape, y.shape) # (856, 144) (856,)

    # 강제로 차원변경 (인자를 1개만 갖고와서 2차원이 생성 3차원으로 강제로 만들어준다...인자가 하나 더 있으면당연 3차원.
    x = x[:, :, np.newaxis]
    y = y.reshape(-1, 1)
    print(x.shape, y.shape) # (856, 144, 1) (856, 1)
    #------------------------------------#
    # mnist minibatch

    hidden_size = 150

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, 1])  #
    ph_y = tf.placeholder(tf.float32)  #

    # cells = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    # outputs, _states = tf.nn.dynamic_rnn(cells, ph_x, dtype=tf.float32)

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
    print('mae:', np.mean(np.abs(preds-y))) # mae: 2.612549101228096

    # shuffle 적용 mae: 0.74608696
    sess.close()


# show_Jena()
rnn_jena()