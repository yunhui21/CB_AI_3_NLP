# Day_25_03_CarEvaluation.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제 1
# car.data 파일을 읽어서
# x_train, x_test, y_train, y_test 데이터를 반환하는 함수를 만드세요

# 문제 2
# 높은 정확도를 갖는 모델을 만드세요


def get_data():
    names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'classes']
    car = pd.read_csv('data/car.data', header=None, names=names)
    car.info()

    enc = preprocessing.LabelEncoder()
    # for col in car.columns:
    #     car[col] = enc.fit_transform(car[col])

    buying   = enc.fit_transform(car.buying)
    maint    = enc.fit_transform(car.maint)
    doors    = enc.fit_transform(car.doors)
    persons  = enc.fit_transform(car.persons)
    lug_boot = enc.fit_transform(car.lug_boot)
    safety   = enc.fit_transform(car.safety)
    classes  = enc.fit_transform(car.classes)

    x = [buying, maint, doors, persons, lug_boot, safety]
    x = np.transpose(x)

    y = classes
    print(x.shape, y.shape)     # (1728, 6) (1728,)
    print(y[:10])

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


def model_car_evaluation_sparse(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = np.max(y_train) + 1     # 4
    w = tf.Variable(tf.random_uniform([n_features, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        # print(i, sess.run(loss, {ph_x: x_train}))

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_sparse(preds_test, y_test)

    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()

results = np.zeros([519, 4])
for i in range(7):
    preds = model_car_evaluation_sparse(x_train, x_test, y_train, y_test)
    results += preds
    # print(preds.shape)

print('-' * 30)
show_accuracy_sparse(results, y_test)
