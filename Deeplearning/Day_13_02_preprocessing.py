# Day_13_02_preprocessing.py
from sklearn import preprocessing, impute
import numpy as np


def add_dummy_feature():
    a = [[1, 3], [5, 7]]
    print(a)

    b = preprocessing.add_dummy_feature(a)
    print(b)
    print(type(b))          # <class 'numpy.ndarray'>
    print(b.dtype)          # float64


def binarizer():
    x = [[-1, 3, -2],
         [5, -7, -4]]

    print(preprocessing.binarize(x))

    bin = preprocessing.Binarizer(threshold=-2)
    bin.fit(x)
    print(bin.transform(x))


def label_binarizer():
    x = [1, 2, 6, 2, 4]

    lb1 = preprocessing.LabelBinarizer()
    lb1.fit(x)

    y = lb1.transform(x)
    print(y)                        # one-hot vector

    print(lb1.classes_)             # [1 2 4 6]

    # 문제
    # 원핫 벡터를 원래 데이터로 복구하세요
    print(np.argmax(y, axis=1))     # [0 1 3 1 2]
    print(lb1.classes_[np.argmax(y, axis=1)])

    print(lb1.inverse_transform(y))
    print('-' * 30)

    labels = ['ok', 'cancel', 'ok']

    lb2 = preprocessing.LabelBinarizer().fit(labels)
    print(lb2.classes_)
    print(lb2.transform(labels))

    labels = ['ok', 'cancel', 'ok', 'apply']

    lb2 = preprocessing.LabelBinarizer().fit(labels)
    print(lb2.classes_)
    print(lb2.transform(labels))

    y = lb2.transform(labels)
    print(lb2.inverse_transform(y))


def label_encoder():
    labels = ['ok', 'cancel', 'ok', 'apply']

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    y = le.transform(labels)
    print(y)
    print(le.classes_)
    print(le.inverse_transform(y))
    print(le.classes_[y])

    # 문제
    # 단순 인코딩된 y를 원핫 벡터로 변환하세요
    n_classes = len(le.classes_)
    onehot = np.eye(n_classes, dtype=np.int32)
    print(onehot)
    print(onehot[y])


def minmax_scale():
    x = [[1, -1, 5],
         [2, 0, -4],
         [0, 1, -10]]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x)

    print(scaler.data_min_)
    print(scaler.data_max_, end='\n\n')

    print(scaler.transform(x), end='\n\n')

    # 문제
    # 아래 수식을 사용해서 minmax 스케일링을 하세요
    # (X - X의 최소값) / (X의 최대값 - X의 최소값)
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)

    print(mn, mx, end='\n\n')
    print((x - mn) / (mx - mn))


def standard_scale():
    x = [[1, -1, 5],
         [2, 0, -4],
         [0, 1, -10]]

    scaler = preprocessing.StandardScaler()
    scaler.fit(x)

    print(scaler.transform(x))


# 결측치 : missing value (na, nan)
def imputer():
    # 4 = (1 + 7) / 2
    # 6 = (2 + 6 + 10) / 3
    x = [[1, 2],
         [np.nan, 6],
         [7, 10]]

    imp = impute.SimpleImputer()
    imp.fit(x)

    print(imp.transform(x))

    x2 = [[np.nan, np.nan],
          [np.nan, np.nan]]

    print(imp.transform(x2))
    print(imp.statistics_)


# add_dummy_feature()
# binarizer()
# label_binarizer()
# label_encoder()
# minmax_scale()
# standard_scale()
imputer()
