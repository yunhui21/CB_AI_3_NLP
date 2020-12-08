# Day_17_02_LinearRegression_cars.py

# 문제
# cars.csv 파일을 학습해서 속도가 30과 50일 때의 제동 거리를 구하세요
# Day_17_01_LinearRegression.py
import tensorflow as tf         # 1.14.0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_xy_1():
    f = open('data/cars.csv', 'r', encoding='utf-8')
    f.readline()

    speeds, distances = [], []
    for line in f:
        items = line.strip().split(',')
        # print(items)

        speeds.append(int(items[1]))
        distances.append(int(items[2]))

    f.close()

    return speeds, distances


def get_xy_2():
    cars = pd.read_csv('data/cars.csv', index_col=0)
    print(cars)
    # print(cars.speed)

    cars.info()

    return cars.speed.values, cars.dist.values


def linear_regression_cars():
    # x, y = get_xy_1()
    x, y = get_xy_2()
    # return
    # x = [1, 2, 3]
    # y = [1, 2, 3]

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)
    # ph_y = tf.placeholder(tf.float32)

    hx = w * ph_x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    # 문제
    # x가 30와 50인 경우에 대해 예측하세요
    # y_hat1, y_hat2 = sess.run(hx, {ph_x: [30, 50]})
    y_hat0, y_hat1, y_hat2 = sess.run(hx, {ph_x: [0, 30, 50]})
    print(y_hat1, y_hat2)
    sess.close()

    # 문제
    # cars.csv 파일을 그래프에 표시하고
    # 위에서 학습한 1차원 방정식을 직선으로 표시하세요
    plt.plot(x, y, 'ro')
    plt.plot([0, 30], [0, y_hat1], 'g')
    plt.plot([0, 30], [y_hat0, y_hat1], 'b')
    plt.show()


linear_regression_cars()









