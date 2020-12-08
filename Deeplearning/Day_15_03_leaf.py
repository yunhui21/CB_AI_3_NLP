# Day_15_03_leaf.py
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm, linear_model, neighbors


# 문제
# leaf.csv 파일을 가져와서
# 70%의 데이터로 학습하고 30%의 데이터에 대해 예측하세요


# 1. 파일 읽기
leaf = pd.read_csv('data/leaf.csv')
print(leaf)

# 2. x, y 데이터 만들기
x = leaf.values[:, 2:]

le = preprocessing.LabelEncoder()
le.fit(leaf.species)

y = le.transform(leaf.species)
print(x.shape, y.shape)
print(y[:5])

# 3. 학습, 검사 데이터 분할하기
data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

# 4. 학습하기
# clf = svm.SVC(gamma=0.001, C=100)
# clf = svm.SVC()
# clf = linear_model.LogisticRegression()
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

# 5. 검사 데이터에 대해 예측하기
print(clf.score(x_test, y_test))

y_hats = clf.predict(x_test)
equals = (y_hats == y_test)
print('acc :', np.mean(equals))
print(y_hats[:5])
