# Day_25_02_BreastCancer.py
# Day_25_01_BreastCancer.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제 1
# breast cancer 파일을 읽어서
# x_train, x_test, y_train, y_test 데이터를 반환하는 함수를 만드세요

# 문제 2
# 93.7% 수준의 정확도를 갖는 모델을 만드세요 (앙상블 사용)


def show_accuracy(preds, labels):
    preds_1 = (preds.reshape(-1) > 0.5)
    y_1 = labels.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(y_1)

    print('acc :', np.mean(preds_1 == y_1))


def get_data():
    #    1. Sample code number            id number
    #    2. Clump Thickness               1 - 10
    #    3. Uniformity of Cell Size       1 - 10
    #    4. Uniformity of Cell Shape      1 - 10
    #    5. Marginal Adhesion             1 - 10
    #    6. Single Epithelial Cell Size   1 - 10
    #    7. Bare Nuclei                   1 - 10
    #    8. Bland Chromatin               1 - 10
    #    9. Normal Nucleoli               1 - 10
    #   10. Mitoses                       1 - 10
    #   11. Class:                        (2 for benign, 4 for malignant)
    names = ['Code', 'Clump', 'Size', 'Shape', 'Adhesion',
             'Epithelial', 'Nuclei', 'Chromatin', 'Nucleoli', 'Mitoses', 'Classes']
    wdbc = pd.read_csv('data/breast-cancer-wisconsin.data', header=None, names=names)
    wdbc.info()

    counts = wdbc.Nuclei.value_counts()
    most_freq = counts.index[0]
    print(counts)
    print(type(most_freq))          # <class 'str'>

    nuclei = wdbc.Nuclei.values

    # 1번
    # print(nuclei[:5])
    # print(set(nuclei))
    #
    # nuclei = ['0' if i == '?' else i for i in nuclei]
    # nuclei = [most_freq if i == '?' else i for i in nuclei]
    # nuclei = np.int64(nuclei)

    # 2번
    # nuclei[[0, 3, 4]] = '0'
    # nuclei[[True, False, False, True]] = '0'
    # nuclei[nuclei == '?'] = '0'
    nuclei[nuclei == '?'] = most_freq
    nuclei = np.int64(nuclei)
    print(set(nuclei))

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(wdbc['Classes'])      # 2, 4

    y = y.reshape(-1, 1)        # (699,) -> (699, 1)
    y = np.float32(y)           # int -> float

    wdbc.drop(['Code', 'Nuclei', 'Classes'], axis=1, inplace=True)

    # 1번
    # wdbc['Nuclei'] = nuclei
    # wdbc.info()
    # x = wdbc.values

    # 2번 코드
    x = np.hstack([wdbc.values, nuclei.reshape(-1, 1)])
    print(x.shape, y.shape)         # (699, 9) (699, 1)

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

        if i % 10 == 0:
            print(i, sess.run(loss, {ph_x: x_train}))

    preds = sess.run(hx, {ph_x: x_test})
    show_accuracy(preds, y_test)

    sess.close()
    return preds


x_train, x_test, y_train, y_test = get_data()
preds = model_wdbc(x_train, x_test, y_train, y_test)

# results = np.zeros(y_test.shape)
# for i in range(7):
#     preds = model_wdbc(x_train, x_test, y_train, y_test)
#     results += preds
#
# print('-' * 30)
# results /= 7
# show_accuracy(results, y_test)
