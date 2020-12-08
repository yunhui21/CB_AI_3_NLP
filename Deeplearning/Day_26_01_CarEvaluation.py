# Day_26_01_CarEvaluation.py
# Day_25_03_CarEvaluation.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제 1
# 25-3 파일을 미니배치로 변환하세요

# 문제 2
# sparse가 아니라 dense(원핫) 형태로 데이터를 변환해서 정확도를 구하세요
# LabelEncoder ==> LabelBinarizer

def get_data():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classes']
    car = pd.read_csv('data/car.data', header=None, names=names)
    car.info()

    # setosa versicolor virginica
    # 0      1          2
    # 1 0 0  0 1 0      0 0 1
    enc = preprocessing.LabelBinarizer()
    # for col in car.columns:
    #     car[col] = enc.fit_transform(car[col])

    buying   = enc.fit_transform(car.buying)        # (1728, 4)
    maint    = enc.fit_transform(car.maint)
    doors    = enc.fit_transform(car.doors)
    persons  = enc.fit_transform(car.persons)
    lug_boot = enc.fit_transform(car.lug_boot)
    safety   = enc.fit_transform(car.safety)
    classes  = enc.fit_transform(car.classes)

    x = np.hstack([buying, maint, doors, persons, lug_boot, safety])

    y = classes
    print(x.shape, y.shape)     # (1728, 21) (1728, 4)
    print(y[:10])

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy_dense(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == y_arg)
    print('acc :', np.mean(equals))


def model_car_evaluation_dense(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    for i in range(epochs):
        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            xx = x_train[n1:n2]
            yy = y_train[n1:n2]

            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})

        # print(i, c / n_iteration)

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_dense(preds_test, y_test)

    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()
model_car_evaluation_dense(x_train, x_test, y_train, y_test)
