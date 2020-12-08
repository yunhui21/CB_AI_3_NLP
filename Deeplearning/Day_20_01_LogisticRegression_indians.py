# Day_20_01_LogisticRegression_indians.py
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd


# 문제
# 피마 인디언 당뇨병 데이터에 대해
# 70%로 학습하고 30%에 대해 정확도를 예측하세요
def logistic_regression_indians():
    indians = pd.read_csv('data/pima-indians.csv')
    print(indians)
    indians.info()

    x = indians.values[:, :-1]
    y = indians.values[:, -1:]

    x = np.float32(x)
    y = np.float32(y)

    # print(type(x), type(y))
    # print(x.shape, y.shape)         # (768, 8) (768, 1)

    data = model_selection.train_test_split(x, y, train_size=0.7)
    x_train, x_test, y_train, y_test = data

    w = tf.Variable(tf.random_uniform([x.shape[1], 1]))     # 8
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    # (768, 1) = (768, 8) @ (8, 1)
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})

        if i % 10 == 0:
            print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = y_test.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    print('acc :', np.mean(preds_1 == y_1))
    sess.close()


# 문제
# 1. 학습(train) 검증(validation) 검사(test)
#    6:2:2로 나눠서 정확도를 구하세요
# 2. 5번 실행한 결과를 평균해서 알려주세요 (65% 이상)
def logistic_regression_indians_validation():
    indians = pd.read_csv('data/pima-indians.csv')

    x = indians.values[:, :-1]
    y = indians.values[:, -1:]

    x = np.float32(x)
    y = np.float32(y)

    # x = preprocessing.minmax_scale(x)
    x = preprocessing.scale(x)

    test_size = int(len(x) * 0.2)

    data = model_selection.train_test_split(x, y, test_size=test_size)
    x_total, x_test, y_total, y_test = data

    results = []
    for i in range(5):
        data = model_selection.train_test_split(x_total, y_total, test_size=test_size)
        x_train, x_valid, y_train, y_valid = data

        # print(x_train.shape, x_valid.shape, x_test.shape)   # (462, 8) (153, 8) (153, 8)
        # print(y_train.shape, y_valid.shape, y_test.shape)   # (462, 1) (153, 1) (153, 1)

        w = tf.Variable(tf.random_uniform([x.shape[1], 1]))     # 8
        b = tf.Variable(tf.random_uniform([1]))

        ph_x = tf.placeholder(tf.float32)

        # (768, 1) = (768, 8) @ (8, 1)
        z = tf.matmul(ph_x, w) + b
        hx = tf.sigmoid(z)

        loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
        loss = tf.reduce_mean(loss_i)

        optimizer = tf.train.GradientDescentOptimizer(0.005)
        train = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            sess.run(train, {ph_x: x_train})

            # if i % 10 == 0:
            #     print(i, sess.run(loss, {ph_x: x_train}))

        preds = sess.run(hx, {ph_x: x_valid})
        preds_1 = (preds.reshape(-1) > 0.5)
        y_1 = y_valid.reshape(-1)

        preds_1 = np.int32(preds_1)
        y_1 = np.int32(y_1)

        acc = np.mean(preds_1 == y_1)
        print('acc :', acc)
        sess.close()

        results.append(acc)

    print('-' * 30)
    print('avg :', np.mean(results))


# logistic_regression_indians()
logistic_regression_indians_validation()




