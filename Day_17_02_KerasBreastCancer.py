# Day_17_02_KerasBreastCancer.py
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
import collections
# 문체
# wdbc  데이터에 대해 70%로 학습하고 30%에 대해 정확도를 구하세요.

# 문제
# 앞에서 풀었던 코드를 소프트맥스 sparse 버전으로 수정하세요.

# 문제
# 앞에서 풀었던 코드를 소프트맥스 dense 버전으로 수정하세요. (원핫 벡터 사용)

def get_data():
    #  1. Sample code number            id number
    #  2. Clump Thickness               1 - 10
    #  3. Uniformity of Cell Size       1 - 10
    #  4. Uniformity of Cell Shape      1 - 10
    #  5. Marginal Adhesion             1 - 10
    #  6. Single Epithelial Cell Size   1 - 10
    #  7. Bare Nuclei                   1 - 10
    #  8. Bland Chromatin               1 - 10
    #  9. Normal Nucleoli               1 - 10
    #  10. Mitoses                       1 - 10
    #  11. Class:                        (2 for benign, 4 fo

    names = ['Clump', 'Size', 'Shape', 'Adhesion', 'Epithelial', 'Nuclie', 'Chromatin', 'Nucleoli', 'Mitoses', 'Class']
    wdbc = pd.read_csv('data/breast-cancer-wisconsin.data', header = None, index_col=0, names= names)
    # breast = np.loadtxt('data/breast-cancer-wisconsin.data')
    print(wdbc)
    wdbc.info()
    print('-'* 30)

    for col in wdbc.columns:
        # print(col, set(wdbc[col]))
        print(col, wdbc[col].unique())
    print('-' * 30)

    freq = wdbc['Nuclie'].value_counts() # 1     402 빈도차가 너무 크니 ?을 1로 바꾸어도 무난하지 않을까.
    print(freq)
    print(freq[0], freq[-1])
    print(freq.index[0])    # 1

    # freq = collections.Counter(wdbc['Nuclie'])
    # print(freq)
    # print(freq.mods_common(3))
    print('-' * 30)

    missing_values = (wdbc.Nuclie == '?') # Series
    # missing_values = (wdbc.Nuclie == '?').values
    print(missing_values)

    nuclie = wdbc.Nuclie.values
    # nuclie[missing_values] = 999
    nuclie[missing_values] = freq.index[0]  # 최빈값
    print(nuclie.dtype)                     # object

    # nuclie = np.float32(nuclie)
    nuclie = nuclie[:, np.newaxis]          # hstack에 사용하기 위해 2차원으로 변환
    print(nuclie.shape)                     # (699, 1)

    x = wdbc.values[:, :-1]
    x = np.hstack([x[:, :6], nuclie, x[:, 7:]])
    print(x.shape, x.dtype)                 # (699, 9) object

    x = np.float32(x)
    y = preprocessing.LabelEncoder().fit_transform(wdbc['Class'])
    y = y[:, np.newaxis]
    print(x.shape, y.shape, x.dtype)                 # (699, 9) (699,) float32

    return model_selection.train_test_split(x, y, train_size=0.7)


def model_breast_cancer():

    x_train, x_test, y_train, y_test = get_data()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[x_train.shape[1]]))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)
    print('loss,acc:', model.evaluate(x_test, y_test, verbose=0))


def model_breast_cancer_softmax_sparse():

    x_train, x_test, y_train, y_test = get_data()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[x_train.shape[1]]))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)
    print('loss,acc:', model.evaluate(x_test, y_test, verbose=0))


def model_breast_cancer_softmax_Dense():

    x_train, x_test, y_train, y_test = get_data()
    # print(y_train[:10])

    # 1번
    # y_train = [(1, 0) if i == 0 else (0, 1) for i in y_train.reshape(-1)]
    # y_test = [(1, 0) if i == 0 else (0, 1) for i in y_test.reshape(-1)]
    # # print(y_train[:10])
    #
    # y_train = np.float32(y_train)
    # y_test  = np.float32(y_test)

    # 2번
    onehot = np.eye(2)
    print(onehot)

    y_train = onehot[y_train.reshape(-1)]
    y_test = onehot[y_test.reshape(-1)]
    print(y_train[:10])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[x_train.shape[1]]))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2)
    print('loss,acc:', model.evaluate(x_test, y_test, verbose=0))

# model_breast_cancer()
# model_breast_cancer_softmax_sparse()
model_breast_cancer_softmax_Dense()
