#
# sklearn, scikit-learn
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

a = [[-2, 2, 5, 6],
     [-4, -7, 2, 0],
     [4, 3, 8, -4]]

# encoding과정에서 사용하는 방식이다.
s1 = preprocessing.MinMaxScaler() # scaling 1th method
s1.fit(a)
b = s1.transform(a)
print(b)
print(s1.fit_transform(a)) # 실제는 데이터셋을 한번에 처리못하는 경우가 있으므로 변수를 두어서 진행한다.
print(preprocessing.minmax_scale(a))
'''
[[0.25 0.9  0.5  1.  ]
 [0.   0.   0.   0.4 ]
 [1.   1.   1.   0.  ]]
'''

s2 = preprocessing.StandardScaler() # Scaling 2en method
s2.fit(a)
b = s2.transform(a)
print(b)
print(s2.fit_transform(a)) # 실제는 데이터셋을 한번에 처리못하는 경우가 있으므로 변수를 두어서 진행한다.
print(preprocessing.scale(a)) # 업계 표준 scale
'''
[[-0.39223227  0.59299945  0.          1.29777137]
 [-0.98058068 -1.4083737  -1.22474487 -0.16222142]
 [ 1.37281295  0.81537425  1.22474487 -1.13554995]]'''

print(np.mean(b), np.mean(b, axis=0))
print(np.std(b), np.std(b, axis=0))

df = pd.read_csv('data/iris(150).csv', index_col=0)
print(df.Species) # series
print(df.Species.values)
print(set(df.Species.values))# {'setosa', 'versicolor', 'virginica'}

# 숫자로 변환 encoding, onehot
e1 = preprocessing.LabelEncoder() # 정확도가 크게 중요하지 않는경우 일반적으로 사용한다.
e1.fit(df.Species)
print(e1.classes_) # ['setosa' 'versicolor' 'virginica']

y = e1.transform(df.Species)
print(y)
print(preprocessing.LabelEncoder().fit_transform(df.Species)) # 한번만 적용할때 쓰는 방법
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]'''
e2 = preprocessing.LabelBinarizer() # feature를 만들때
b = e2.fit_transform(df.Species)
print(b) # 0,1 구성된 데이터로 변환  [[0 0 1] [0 0 1]]
print(e2.classes_) # ['setosa' 'versicolor' 'virginica']

c = np.argmax(b, axis=1)
print(c)               #
print(e2.classes_[c])  # decoding 과정

d = np.eye(len(e2.classes_), dtype=np.int32)
print(d) # [[1 0 0] [0 1 0] [0 0 1]]
print(d[c]) # labelEncoder 값과 같다.

x = df.values[:, :-1]
y = e1.transform(df.values[:, -1]) # (150, 4) (150,)
# y = e1.transform(df.values[:, -1:]) # error
y = y.reshape(-1, 1) # (150, 4) (150, 1)
print(x.shape, y.shape)

# 문제
# 학습과 검사 데이터로 나누세요. (7:3)

# 수도으로 작업
# train_size =len(x) *0.7  # 105.0
train_size =int(len(x) *0.7)  #  105
print(train_size)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(x_train.shape, x_test.shape) # (105, 4) (45, 4)
print(y_train.shape, y_test.shape) # (105, 1) (45, 1)
print(y_train[:10]) # 2차원
print(y_train[:10].reshape(-1)) # [0 0 0 0 0 0 0 0 0 0]

# 자동으로 나누면서 shuffle을 같이 한다.
data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test,y_train, y_test = data

print(x_train.shape, x_test.shape) # (105, 4) (45, 4)
print(y_train.shape, y_test.shape) # (105, 1) (45, 1)
print(y_train[:10]) # 2차원
print(y_train[:10].reshape(-1)) # [0 0 2 1 0 0 0 2 0 2]
