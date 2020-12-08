# Day_23_03_EnsembleIndian.py
import tensorflow as tf
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd


# 문제
# 피마 인디언 당뇨병 데이터에 대해
# 70%로 학습하고 30%에 대해 정확도를 예측하세요
# (앙상블 모델을 구축하세요. 정확도는 70% 이상이면 좋겠습니다)

# 문제
# 지금까지 배운 모든 것들을 활용해서 정확도를 70% 이상으로 끌어올려보세요

def show_accuracy(preds, labels):
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = labels.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    print('acc :', np.mean(preds_1 == y_1))


def get_data():
    indians = pd.read_csv('data/pima-indians.csv')

    x = indians.values[:, :-1]
    y = indians.values[:, -1:]

    # x = preprocessing.minmax_scale(x)
    x = preprocessing.scale(x)

    x = np.float32(x)
    y = np.float32(y)

    return model_selection.train_test_split(x, y, train_size=0.7)


def logistic_regression_indians(x_train, x_test, y_train, y_test):
    # w = tf.Variable(tf.random_uniform([x_train.shape[1], 1]))     # 8
    # b = tf.Variable(tf.random_uniform([1]))
    name = str(np.random.rand())
    w = tf.get_variable(name, shape=[x_train.shape[1], 1], initializer=tf.glorot_uniform_initializer)
    b = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)

    # (768, 1) = (768, 8) @ (8, 1)
    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.0001)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})

        # if i % 10 == 0:
        #     print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    show_accuracy(preds, y_test)

    sess.close()
    return preds


x_train, x_test, y_train, y_test = get_data()

results = np.zeros(y_test.shape)
for i in range(7):
    preds = logistic_regression_indians(x_train, x_test, y_train, y_test)
    results += preds
    # print(preds.shape)

print('-' * 30)
results /= 7
show_accuracy(results, y_test)
print(results[:3])

