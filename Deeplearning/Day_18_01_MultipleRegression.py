# Day_18_01_MultipleRegression.py
import tensorflow as tf


# 문제
# 피처를 두 개로 늘리세요(공부한 시간, 출석한 일수)
def multiple_regression_1():
    #       1         1        0
    # hx = w1 * x1 + w2 * x2 + b
    #  y = x1 + x2
    x1 = [1, 0, 3, 0, 5]        # 공부한 시간
    x2 = [0, 2, 0, 4, 0]        # 출석한 일수
    y = [1, 2, 3, 4, 5]         # 성적

    w1 = tf.Variable(tf.random_uniform([1]))
    w2 = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w1 * x1 + w2 * x2 + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# x1과 x2를 변수 1개로 만드세요
def multiple_regression_2():
    x = [[1, 0, 3, 0, 5],       # 공부한 시간
         [0, 2, 0, 4, 0]]       # 출석한 일수
    y = [1, 2, 3, 4, 5]         # 성적

    w = tf.Variable(tf.random_uniform([2]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w[0] * x[0] + w[1] * x[1] + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# bias를 없애보세요
def multiple_regression_3():
    # x = [[1, 0, 3, 0, 5],       # 공부한 시간
    #      [0, 2, 0, 4, 0],       # 출석한 일수
    #      [1, 1, 1, 1, 1]]
    # x = [[1, 0, 3, 0, 5],       # 공부한 시간
    #      [1, 1, 1, 1, 1],
    #      [0, 2, 0, 4, 0]]       # 출석한 일수
    x = [[1, 1, 1, 1, 1],
         [1, 0, 3, 0, 5],       # 공부한 시간
         [0, 2, 0, 4, 0]]       # 출석한 일수
    y = [1, 2, 3, 4, 5]         # 성적

    w = tf.Variable(tf.random_uniform([3]))
    # b = tf.Variable(tf.random_uniform([1]))

    # hx = w[0] * x[0] + w[1] * x[1] + b
    # hx = w[0] * x[0] + w[1] * x[1] + w[2]
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * 1
    hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train)
        print(i, sess.run(loss))

    print(sess.run(w))
    sess.close()


# 문제
# hx 연산을 행렬 곱셈으로 바꾸세요 (tf.matmul)
def multiple_regression_4():
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3]))

    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # hx = w @ x
    # () = (1, 3) @ (3, 5)
    hx = tf.matmul(w, x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    sess.close()


# 문제
# 3시간 공부하고 4번 출석한 학생과
# 5시간 공부하고 1번 출석한 학생의 성적을 구하세요 (placeholder)
def multiple_regression_5():
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3]))
    ph_x = tf.placeholder(tf.float32)

    # (1, 5) = (1, 3) @ (3, 5)
    hx = tf.matmul(w, ph_x)

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print(sess.run(hx, {ph_x: x}))
    # [[1.2553192 2.1303844 3.071868  3.9137244 4.8884172]]

    print(sess.run(hx, {ph_x: [[1., 1., 1., 1., 1.],
                               [1., 0., 3., 0., 5.],
                               [0., 2., 0., 4., 0.]]}))

    print(sess.run(hx, {ph_x: [[1., 1.],
                               [3., 5.],
                               [4., 1.]]}))
    sess.close()


# 문제
# 행렬 곱셈에서 w와 x의 위치를 바꾸세요
# 문제
# 3시간 공부하고 4번 출석한 학생과
# 5시간 공부하고 1번 출석한 학생의 성적을 구하세요 (placeholder)
def multiple_regression_6():
    # x = [[1., 1., 1., 1., 1.],
    #      [1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.]]
    x = [[1., 1., 0.],
         [1., 0., 2.],
         [1., 3., 0.],
         [1., 0., 4.],
         [1., 5., 0.]]
    y = [[1],
         [2],
         [3],
         [4],
         [5]]

    w = tf.Variable(tf.random_uniform([3, 1]))
    ph_x = tf.placeholder(tf.float32)

    # (5, 1) = (5, 3) @ (3, 1)
    hx = tf.matmul(ph_x, w)

    #     (5, 1) - (1, 5)
    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))

    print(sess.run(hx, {ph_x: [[1., 3., 4.],
                               [1., 5., 1.]]}))
    sess.close()


# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
# multiple_regression_5()
multiple_regression_6()


# import numpy as np
#
# a = np.arange(5)
# b = 2
# c = [b, b, b, b, b]
#
# print(a + b)
# print(a + c)



