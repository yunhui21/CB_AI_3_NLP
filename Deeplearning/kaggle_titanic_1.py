# kaggle_titanic_1.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제
# 에러 없이 동작하는 모델을 만들어서 결과를 리더보드에 등록하세요
# 1. 모델 만들기
# 2. 서브미션 파일 만들기


def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv', index_col=0)
    print(titanic, end='\n\n')

    titanic.drop(['Age', 'Cabin', 'Embarked'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    titanic.info()

    x = titanic.values[:, 1:]
    y = titanic.values[:, :1]

    return x, np.float32(y)


def get_data_test():
    titanic = pd.read_csv('kaggle/titanic_test.csv', index_col=0)

    titanic.drop(['Age', 'Cabin', 'Embarked'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    x = titanic.values
    ids = titanic.index.values

    return x, ids


def make_submission(ids, preds):
    f = open('kaggle/titanic_submission.csv', 'w', encoding='utf-8')

    print('PassengerId,Survived', file=f)

    for i in range(len(ids)):
        result = int(preds[i] > 0.5)
        print('{},{}'.format(ids[i], result), file=f)

    f.close


def model_titanic(x_train, x_test, y_train):
    w = tf.Variable(tf.random_uniform([x_train.shape[1], 1]))
    b = tf.Variable(tf.random_uniform([1]))

    ph_x = tf.placeholder(tf.float32)

    z = tf.matmul(ph_x, w) + b
    hx = tf.sigmoid(z)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.01)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})

    sess.close()
    return preds.reshape(-1)


x_train, y_train = get_data_train()
x_test, ids = get_data_test()

print(x_train.shape, x_test.shape, y_train.shape)   # (891, 4) (418, 4) (891, 1)

preds = model_titanic(x_train, x_test, y_train)
make_submission(ids, preds)


