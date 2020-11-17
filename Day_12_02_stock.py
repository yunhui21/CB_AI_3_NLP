# Day_12_02_stock.py
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing, model_selection

np.set_printoptions(linewidth=1000)

# 문제
# 주식 데이터를 읽어서 1주일 단위로 학습해서 그 다음 날 종가를 예측하세요
# (예측 결과는 그래프로 표시합니다)


def make_data(seq_len):
    # 문제
    # x와 y 데이터를 만드세요.
    # stock = pd.read_csv('data/stock_daily.csv') # #처리된 줄을 읽어 들이지 못한다.
    stock = np.loadtxt('data/stock_daily.csv', delimiter=',') # numpy에서는 #을 보면 주석으로 받아들인다.
    print(stock)

    rng = [i for i in range(len(stock) - seq_len )] # +1 삭제(8번째가 y라서)

    x = [stock[s:s+seq_len] for s in rng]
    # y = [stock[s+seq_len][-1] for s in rng] # 마지막 거래가만 필요하다.
    y = [stock[s+seq_len, -1:] for s in rng] # 마지막 거래가만 필요하다. 2차원의 값을 가져야 한다.

    print(y[:3]) # [809.559998, 808.380005, 806.969971]

    x = np.float32(x)
    y = np.float32(y)
    print(x.shape, y.shape) # (725, 7, 5) (725, 1)

    return model_selection.train_test_split(x, y, train_size=0.7)
    # regression

def rnn_stock_1():
    seq_len, n_features = 7, 5
    x_train, x_test, y_train, y_test = make_data(seq_len)

    # regresson mini_batch_size 적용하지 않는다.

    ph_x = tf.placeholder(tf.float32, shape=[None, seq_len, n_features])

    hidden_size = 15
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)
    print(outputs.shape) # (?, 7, 15)
    #                           (?, 7, 15) seq_len의 마지막값
    hx = tf.layers.dense(outputs[:, -1, :], units= 1, activation=None) # 종가 1개의 값

    loss_i = (hx - y_train) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.compat.v1.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    # 문제
    # 정답과 예측 결과를 그래프로 그려보세요.
    # 정답은 빨강, 예측은 초록
    preds = sess.run(hx, {ph_x: x_test})
    # print(preds.shape) # (218, 1)
    preds = preds.reshape(-1)
    ytest = y_test.reshape(-1)

    plt.plot(range(len(ytest)), ytest, 'r')
    plt.plot(range(len(ytest)), preds, 'g')
    plt.show()

    sess.close()


rnn_stock_1()




