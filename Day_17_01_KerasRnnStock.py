# Day_12_03_stock.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing, model_selection
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)

# tensorflow
# x: tenso ensor nsorf
# y: ensor nsorf

# 시계열
# x: tenso ensor nsorf
# y:     r     f

# 문제
# stock_daily.csv 파일을 읽어서 x, y 데이터로 분리하세요

# 문제
# 주식 데이터를 7:3으로 나눠서 예측 결과를 정답과 함께 그래프로 그려주세요


def make_data(seq_len):
    stock = pd.read_csv('data/GOOG.csv', index_col=0)       # 부적격
    print(stock)

    # stock = np.loadtxt('data/GOOS.csv', delimiter=',')

    stock = stock.values
    stock = np.hstack([stock[:, :3], stock[:, -1:], stock[:, 3:4]])
    print(stock.shape)

    stock = preprocessing.scale(stock)

    rng = range(len(stock) - seq_len)

    x = [stock[s:s+seq_len] for s in rng]       # [0, 7) [1, 8) [2, 9)
    # y = [stock[s+seq_len][-1] for s in rng]   #  7      8      9
    y = [stock[s+seq_len, -1:] for s in rng]    # :을 붙여서 2차원 추출

    x = np.float32(x)
    y = np.float32(y)
    print(x.shape, y.shape)     # (725, 7, 5) (725, 1)

    return model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)


def rnn_stock(seq_len, loop_count):
    x_train, x_test, y_train, y_test = make_data(seq_len)

    _, seq_len, n_features = x_train.shape
    hidden_size = 21

    # ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_features])
    #
    # cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size) for _ in range(2)]
    # multi = tf.nn.rnn_cell.MultiRNNCell(cells)
    # outputs, _states = tf.nn.dynamic_rnn(multi, ph_x, dtype=tf.float32)
    # print(outputs.shape)            # (?, 7, 21)
    # print(outputs[:, -1, :].shape)  # (?, 21)
    #
    # # ---- #
    # hx = tf.layers.dense(outputs[:, -1, :], 1, activation=None)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[seq_len, n_features]))
    model.add(tf.keras.layers.SimpleRNN(hidden_size, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation=None))
    model.summary()
    # return

    # loss_i = (hx - y_train) ** 2        # mse: mean square error
    # loss = tf.reduce_mean(loss_i)
    # optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    # train = optimizer.minimize(loss)

    model.compile(optimizer= tf.keras.optimizers.Adam(lr=0.01),
                  loss = tf.keras.losses.mse)
    # sess = tf.compat.v1.Session()
    # sess.run(tf.compat.v1.global_variables_initializer())
    #
    # for i in range(loop_count):
    #     sess.run(train, {ph_x: x_train})
    #     print(i, sess.run(loss, {ph_x: x_train}))

    model.fit(x_train, y_train, epochs=1, batch_size= len(x_train), verbose=2)
    # preds = sess.run(hx, {ph_x: x_test})
    # print(preds.shape, y_test.shape)        # (218, 1) (218, 1)
    #
    # sess.close()
    preds = model.predict(x_test)
    print(preds.shape, y_test.shape)

    plt.plot(y_test.reshape(-1), 'r', label='label')
    plt.plot(preds.reshape(-1), 'g', label='preds')
    plt.legend()
    plt.xlabel('prediction')
    plt.show()


rnn_stock(seq_len=7, loop_count=300)

# (3, 5, 2)
# (*, 5, 2)
# (3, *, 2)
# (3, 5, *)