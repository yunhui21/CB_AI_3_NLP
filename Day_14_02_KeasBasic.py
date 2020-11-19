# Day_14_02_KeasBasic.py
import tensorflow as tf
import pandas as pd
import numpy as np
# 문제
# jena 파일에 있는 온도 데이터를 리니어 리그레션을 사용해서 평균 오차를 계산하세요.

# 문제
# trees.csv 파일로 학습을 해서
# Girth가 10, Height가 75일 때와
# Girth가 15, Height가 80일 때의 Volum을 예측하세요.


def linear_regression():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.mse)

    # model.fit(x, y)
    model.fit(x, y, epochs=100, verbose=0)
    print(model.evaluate(x, y, verbose=0))
    print(model.predict(x))


def linear_regression_jena():
    jena = pd.read_csv('data/jena_climate_2009_2016.csv', index_col=0)
    degc = jena['T (degC)'].values
    degc = degc[:1000]

    x, y = degc[:-144], degc[144:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100, verbose=0)
    print(model.evaluate(x, y, verbose=0))
    # print(model.predict(x))
    preds = model.predict(x) # (856, 1)
    print(preds.shape)
    print(y.shape)
    print('mae:', np.mean(np.abs(preds.reshape(-1)-y))) # mae: 4.824650473048754

    # mae를 metric에 전달하는 방법------------------------------------------------

# https://forge.scilab.org/index.php/p/rdataset/source/tree/master/csv/datasets/trees.csv
# tree csv 검색
def multiple_regression_trees():
    trees = pd.read_csv('data/trees.csv', index_col=0)
    # print(trees)
    x = np.transpose([trees['Girth'],trees['Height']])

    y = np.reshape(trees['Volume'].values, newshape=[-1,1])

    print(x.shape, y.shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.mse)

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # print(x[:2])
    xx = [[10, 75],
          [15, 80]]

    preds = model.predict(xx)  # (856, 1)
    print(preds)
    print(preds.reshape(-1))

# linear_regression()
# linear_regression_jena()
multiple_regression_trees()