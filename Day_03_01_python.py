# Day_03_01_python.py
import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np


# 문제
# iris 파일을 읽어서 화면에 출력하세요 (pandas 사용)
# df = pd.read_csv('data/iris(150).csv')
df = pd.read_csv('data/iris(150).csv', index_col=0)
print(df)

print(df.index)
print(df.columns)
print(df.values)
print(type(df.values))
print()

# 문제
# 데이터프레임(df)을 x와 y 데이터로 분할하세요
x = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = df[['Species']]
print(x.shape, y.shape)         # (150, 4) (150, 1)

df_x = df.iloc[:, :-1]
df_y = df.iloc[:, -1]
print(df_x.shape, df_y.shape)   # (150, 4) (150,)

x = df.values[:, :-1]
y = df.values[:, -1:]
print(x.shape, y.shape)         # (150, 4) (150, 1)
print()

# df.drop(['Species'], axis=1, inplace=True)
# print(df)

y = df.Species
y = df.Species.values
y = y.reshape(-1, 1)
# y = df['Species']
df2 = df.drop(['Species'], axis=1)
print(df2)

x = df2.values
print(x.shape, y.shape)
print('-' * 30)

a = [[-2, 1, 5, 6],
     [-4, -7, 2, 0],
     [4, 3, 8, -4]]

s1 = preprocessing.MinMaxScaler()
s1.fit(a)

b = s1.transform(a)
print(b)
print(s1.fit_transform(a))
print(preprocessing.MinMaxScaler().fit_transform(a))
print(preprocessing.minmax_scale(a))

# zero sum, std 1.0
s2 = preprocessing.StandardScaler()
s2.fit(a)

b = s2.transform(a)
print(b)
print(s2.fit_transform(a))
print(preprocessing.StandardScaler().fit_transform(a))
print(preprocessing.scale(a))
print()

print(np.sum(b), np.sum(b, axis=0))
# print(np.mean(b), np.mean(b, axis=0))
print(np.std(b), np.std(b, axis=0))
print('-' * 30)

e1 = preprocessing.LabelEncoder()
print(e1.fit_transform(df.Species))
print(e1.fit_transform(df.Species.values))
# print(e1.fit_transform(df.Species.values.reshape(-1, 1)))     # 경고

e2 = preprocessing.LabelBinarizer()
c = e2.fit_transform(df.Species)

print(c)
print(e2.classes_)

d = np.argmax(c, axis=1)
print(d)

e = np.eye(len(e2.classes_), dtype=np.int32)
print(e)
print(e[d])

# s1: minmax scale
print(s1.data_max_)
print(s1.data_min_)

f = np.array([-3, -1, 0, 2])
print((f - -4) / (4 - -4))
print((f - s1.data_min_) / (s1.data_max_ - s1.data_min_))

print('-' * 30)

data = df.values
print(data.shape)           # (150, 5)

# 문제
# x, y를 학습과 검사 데이터로 나누세요(70%, 30%)
x = data[:, :-1]
y = data[:, -1:]
print(x.shape, y.shape)

train_size = int(len(x) * 0.7)
print(train_size)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 75:25
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7)
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
