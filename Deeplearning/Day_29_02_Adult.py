# Day_29_02_Adult.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제 1
# 연소득 데이터에 대해 정확도를 구하세요 (83% 이상, 앙상블 금지)
# Day_28_01_BankMarketing.py

def get_data_sparse(file_path):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'income']
    adult = pd.read_csv(file_path, header=None, names=names)

    # print(adult, end='\n\n')
    # print(adult.describe(), end='\n\n')
    # adult.info()

    age = adult.age.values
    education_num = adult.education_num.values
    fnlwgt = adult.fnlwgt.values
    capital_gain = adult.capital_gain.values
    capital_loss = adult.capital_loss.values
    hours_per_week = adult.hours_per_week.values

    enc = preprocessing.LabelEncoder()

    workclass = enc.fit_transform(adult.workclass)
    education = enc.fit_transform(adult.education)
    marital = enc.fit_transform(adult.marital)
    occupation = enc.fit_transform(adult.occupation)
    relationship = enc.fit_transform(adult.relationship)
    race = enc.fit_transform(adult.race)
    sex = enc.fit_transform(adult.sex)
    native_country = enc.fit_transform(adult.native_country)

    y = enc.fit_transform(adult.income)
    y = y.reshape(-1, 1)
    y = np.float32(y)

    x = np.transpose([
        age, education_num, fnlwgt, capital_gain, capital_loss, hours_per_week,
        workclass, education, marital, occupation, relationship, race, sex, native_country
    ])

    x = preprocessing.scale(x)

    return model_selection.train_test_split(x, y, train_size=0.7)


def get_data_dense(file_path):
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'income']
    adult = pd.read_csv(file_path, header=None, names=names)

    age = adult.age.values
    education_num = adult.education_num.values
    fnlwgt = adult.fnlwgt.values
    capital_gain = adult.capital_gain.values
    capital_loss = adult.capital_loss.values
    hours_per_week = adult.hours_per_week.values

    enc = preprocessing.LabelBinarizer()

    workclass = enc.fit_transform(adult.workclass)
    education = enc.fit_transform(adult.education)
    marital = enc.fit_transform(adult.marital)
    occupation = enc.fit_transform(adult.occupation)
    relationship = enc.fit_transform(adult.relationship)
    race = enc.fit_transform(adult.race)
    sex = enc.fit_transform(adult.sex)
    native_country = enc.fit_transform(adult.native_country)

    y = preprocessing.LabelEncoder().fit_transform(adult.income)
    y = y.reshape(-1, 1)
    y = np.float32(y)

    columns = np.transpose([age, education_num, fnlwgt, capital_gain, capital_loss, hours_per_week])
    x = np.hstack([columns, workclass, education, marital, occupation, relationship, race, sex, native_country])

    x = preprocessing.scale(x)

    return model_selection.train_test_split(x, y, train_size=0.7)


def show_accuracy(preds, labels):
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = labels.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    # print('acc :', np.mean(preds_1 == y_1))
    return np.mean(preds_1 == y_1)


def model_adult(x_train, x_test, y_train, y_test):
    w1 = tf.get_variable('w1', shape=[x_train.shape[1], 64],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([64]))

    w2 = tf.get_variable('w2', shape=[64, 64],
                         initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([64]))

    w3 = tf.get_variable('w3', shape=[64, 32],
                         initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([32]))

    w4 = tf.get_variable('w4', shape=[32, 16],
                         initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([16]))

    w5 = tf.get_variable('w5', shape=[16, 1],
                         initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)       # drop-out

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_prob=ph_d)
    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_prob=ph_d)
    z3 = tf.matmul(d2, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_prob=ph_d)
    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)
    d4 = tf.nn.dropout(r4, keep_prob=ph_d)
    z5 = tf.matmul(d4, w5) + b5
    hx = tf.sigmoid(z5)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z5)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100
    batch_size = 32
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        np.random.shuffle(indices)

        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy, ph_d: 0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d: 0.7})

        # print(i, c / n_iteration)

        preds = sess.run(hx, {ph_x: x_test, ph_d: 1.0})
        avg = show_accuracy(preds, y_test)

        print('{:3} : {:7.5f}  {:7.5f}'.format(i, c / n_iteration, avg))

    sess.close()


# x_train, x_test, y_train, y_test = get_data_sparse('data/adult.data')  # (22792, 14) (9769, 14) (22792, 1) (9769, 1)
x_train, x_test, y_train, y_test = get_data_dense('data/adult.data')   # (22792, 107) (9769, 107) (22792, 1) (9769, 1)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

model_adult(x_train, x_test, y_train, y_test)
