# Day_26_02_Abalone.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd

np.set_printoptions(linewidth=1000)

# 문제 1
# 싱글 모델을 사용해서 정확도를 예측하세요 (앙상블 금지)
# get_data 함수를 만들어서 사용합니다

# 문제 2
# 클래스의 갯수를 3개로 축약하세요 (0~9, 10~19, 20~29)


def get_data():
    names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
    abalone = pd.read_csv('data/abalone.data', header=None, names=names)
    print(abalone)

    enc = preprocessing.LabelEncoder()
    gender = enc.fit_transform(abalone.Sex)
    y = abalone.Rings.values // 10

    print(np.unique(y))
    print(np.unique(enc.fit_transform(y)))
    # y = enc.fit_transform(y)
    # exit(-1)

    abalone.drop(['Sex', 'Rings'], axis=1, inplace=True)
    x = np.hstack([gender.reshape(-1, 1), abalone.values])      # (4177, 8) (4177,)

    print(x.shape, y.shape)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


def model_abalone_sparse(x_train, x_test, y_train, y_test):
    n_features = x_train.shape[1]
    n_classes = np.max(y_train) + 1     # 30
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

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        # print(i, sess.run(loss, {ph_x: x_train}))

    preds_test = sess.run(hx, {ph_x: x_test})
    show_accuracy_sparse(preds_test, y_test)

    sess.close()
    return preds_test


x_train, x_test, y_train, y_test = get_data()
# model_abalone_sparse(x_train, x_test, y_train, y_test)

results = np.zeros([len(x_test), np.max(y_train) + 1])
preds_arg_list = []
for i in range(7):
    preds = model_abalone_sparse(x_train, x_test, y_train, y_test)
    results += preds
    # print(preds.shape)

    preds_arg_list.append(np.argmax(preds, axis=1))

print('-' * 30)
show_accuracy_sparse(results, y_test)
print()

for preds_arg in preds_arg_list:
    print(preds_arg[:30])

print('-' * 50)
print(y_test[:30])










