# Day_25_01_BreastCancer.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제 1
# wdbc.data 파일을 읽어서
# x_train, x_test, y_train, y_test 데이터를 반환하는 함수를 만드세요

# 문제 2
# 97.5% 수준의 정확도를 갖는 모델을 만드세요 (앙상블 사용 금지)

# 문제 3
# 앙상블 모델로 변환하세요


def show_accuracy(preds, labels):
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = labels.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    print('acc :', np.mean(preds_1 == y_1))


def get_data():
    wdbc = pd.read_csv('data/wdbc.data', header=None)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(wdbc[1])
    # x = wdbc[wdbc.columns[2:]]
    # x = wdbc[range(2, 32)]

    y = y.reshape(-1, 1)        # (569,) -> (569, 1)
    y = np.float32(y)           # int -> float

    print(wdbc[1].values[:5])
    print(y[:5])

    wdbc.drop([0, 1], axis=1, inplace=True)
    x = wdbc.values

    print(x.shape, y.shape)         # (569, 30) (569,)

    return model_selection.train_test_split(x, y, train_size=0.7)


def model_wdbc(x_train, x_test, y_train, y_test):
    w = tf.Variable(tf.random_uniform([x_train.shape[1], 1]))
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.0001)
    optimizer = tf.train.AdamOptimizer(0.01)
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
    preds = model_wdbc(x_train, x_test, y_train, y_test)
    results += preds

print('-' * 30)
results /= 7
show_accuracy(results, y_test)
