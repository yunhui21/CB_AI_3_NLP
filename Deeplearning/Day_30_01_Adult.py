# Day_30_01_Adult.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd

np.set_printoptions(linewidth=1000)

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

    return x, y


def show_accuracy(preds, labels):
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = labels.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    # print('acc :', np.mean(preds_1 == y_1))
    return np.mean(preds_1 == y_1)


def model_adult(x_train, x_test, y_train, y_test):
    w1 = tf.get_variable('w1', shape=[x_train.shape[1], 41],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([41]))

    w2 = tf.get_variable('w2', shape=[41, 31],
                         initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([31]))

    w3 = tf.get_variable('w3', shape=[31, 21],
                         initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([21]))

    w4 = tf.get_variable('w4', shape=[21, 11],
                         initializer=tf.glorot_uniform_initializer)
    b4 = tf.Variable(tf.zeros([11]))

    w5 = tf.get_variable('w5', shape=[11, 1],
                         initializer=tf.glorot_uniform_initializer)
    b5 = tf.Variable(tf.zeros([1]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)
    z3 = tf.matmul(r2, w3) + b3
    r3 = tf.nn.relu(z3)
    z4 = tf.matmul(r3, w4) + b4
    r4 = tf.nn.relu(z4)
    z5 = tf.matmul(r4, w5) + b5
    hx = tf.sigmoid(z5)

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z5)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 3
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

            sess.run(train, {ph_x: xx, ph_y: yy})
            c += sess.run(loss, {ph_x: xx, ph_y: yy})

        # print(i, c / n_iteration)

        preds = sess.run(hx, {ph_x: x_test})
        avg = show_accuracy(preds, y_test)

        print('{:3} : {:7.5f}  {:7.5f}'.format(i, c / n_iteration, avg))

    sess.close()

# 문제 1
# train과 test 데이터프레임의 다른 점을 찾아보세요

def show_difference():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'income']
    train = pd.read_csv('data/adult.data', header=None, names=names)
    test = pd.read_csv('data/adult.test', header=None, names=names)

    # train.info()

    # 107 vs. 106
    # LabelBinerizer.fit_transform 함수에서 만들어낸 원핫 벡터의 갯수가 다르기 때문.
    # train.grade : a, b, c, d, e
    #  test.grade : a, b, d, e

    # 1. 문제를 읽으킬 수 있는 가능성이 있는 컬럼 조사 (object)
    # 2. 각 컬럼에 들어있는 유니크한 데이터 비교 (train/test)

    # print(train.workclass.unique())
    # print(test.workclass.unique())

    # print(sorted(train.workclass.unique()))
    # print(sorted(test.workclass.unique()))

    for col in names:
        # print(train[col].dtype)

        if train[col].dtype == np.object:
            # if len(train[col].unique()) == len(test[col].unique()):
            #     continue
            #
            # print(col)
            # print(sorted(train[col].unique()))
            # print(sorted(test[col].unique()))
            #
            # print(len(train[col].unique()))
            # print(len(test[col].unique()))

            # 집합 : 합집합, 여집합, 차집합
            s1 = set(train[col])
            s2 = set(test[col])

            if not (s1 - s2):
                continue

            # print(s1)
            # print(s2)
            print(s1 - s2)      # 차집합
            # print(s2 - s1)    # 동작 안함
            # print()

    print('-' * 30)

    enc = preprocessing.LabelBinarizer()
    enc.fit(test.native_country)

    native_country = enc.transform(test.native_country)
    print(native_country.shape)

    print(enc.classes_)

    countries = list(enc.classes_) + [' Holand-Netherlands']
    countries = sorted(countries)

    enc.classes_ = np.array(countries)

    # 에러
    # enc.classes_.append(' Holand-Netherlands')

    native_country = enc.transform(test.native_country)
    print(native_country.shape)


def find_missing_values():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'income']
    # train = pd.read_csv('data/adult.data', header=None, names=names, delimiter=', ')
    train = pd.read_csv('data/adult.data', header=None, names=names)
    # train.info()

    bools = (train.workclass == ' ?')
    # bools = (train.workclass.values == '?')
    print(bools)
    print(bools.sum())
    print(np.sum(bools))
    print('-' * 30)

    # filtered = train[bools]
    filtered = train[train.workclass != ' ?']
    print(filtered)
    print('-' * 30)

    # 문제
    # 결측치가 포함된 컬럼을 찾아보세요
    for col in train.columns:
        if train[col].dtype == np.object:
            na_sum = np.sum(train[col] == ' ?')
            print('{:15} : {}'.format(col, na_sum))


# 에러나는 코드. 수정하지 않았음.
# x_train, y_train = get_data_dense('data/adult.data')   # (32561, 107) (32561, 1)
# x_test, y_test = get_data_dense('data/adult.test')     # (16281, 106) (16281, 1)
#
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
#
# model_adult(x_train, x_test, y_train, y_test)

# show_difference()
find_missing_values()
