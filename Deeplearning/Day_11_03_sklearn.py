# Day_11_03_sklearn.py
from sklearn import datasets


def basic_1():
    iris = datasets.load_iris()
    print(type(iris))           # <class 'sklearn.utils.Bunch'>
    print(iris.keys())
    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'frame'])

    print(iris['feature_names'])
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    print(iris['target_names'])
    # ['setosa' 'versicolor' 'virginica']

    print(iris['data'][:5])
    # [[5.1 3.5 1.4 0.2]
    #  [4.9 3.  1.4 0.2]
    #  [4.7 3.2 1.3 0.2]
    #  [4.6 3.1 1.5 0.2]
    #  [5.  3.6 1.4 0.2]]

    print(type(iris['data']))                       # <class 'numpy.ndarray'>
    print(iris['data'].shape, iris['data'].dtype)   # (150, 4) float64

    print(iris['target'])
    # [0 0 0 0 0 ... 2 2]

    print(iris['DESCR'])


# 문제
# digits 데이터셋에 대해 조사하세요
def basic_2():
    digits = datasets.load_digits()
    print(digits.keys())
    # dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

    print(digits['data'].shape)         # (1797, 64)
    print(digits['images'].shape)       # (1797, 8, 8)

    print(digits['data'][0])
    print(digits['images'][0])
    print(digits['target'][0])


# basic_1()
basic_2()


