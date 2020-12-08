# Day_19_01_MultipleRegression_boston.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection

# xlrd 설치하고 BostonHousing.xls 다운로드

# 문제 1
# 보스턴 주택가격 파일을 읽어오세요 (팬다스 사용)
boston = pd.read_excel('data/BostonHousing.xls')
# print(boston)

# 문제 2
# 보스턴 데이터를 x, y로 변환하세요
print(boston.values.shape)

# x = boston.values[:, :-1]     # (506, 13)
# y = boston.values[:, -1:]     # (506, 1)

y = boston.MEDV.values          # (506,)
y = y.reshape(-1, 1)            # (506, 1)

x = boston.drop(['MEDV'], axis=1).values
print(x.shape, y.shape)

# 문제 3
# 보스턴 데이터에 대해 학습하세요

# 문제 4
# 마지막 1개를 제외한 데이터로 학습하고 마지막 1개에 대해 결과를 알려주세요
# x_train, x_test = x[:-1], x[-1:]
# y_train, y_test = y[:-1], y[-1:]

# 문제 5
# 70% 데이터로 학습하고 30% 데이터로 예측하고
# 평균 오차가 얼마나 되는지 알려주세요
data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

w = tf.Variable(tf.random_uniform([x.shape[1], 1]))     # 13
ph_x = tf.placeholder(tf.float32)

# (506, 1) = (506, 13) @ (13, 1)
hx = tf.matmul(ph_x, w)

loss_i = (hx - y_train) ** 2
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    sess.run(train, {ph_x: x_train})
    print(i, sess.run(loss, {ph_x: x_train}))

preds = sess.run(hx, {ph_x: x_test})
print(preds.shape)

preds_1 = preds.reshape(-1)
ytest_1 = y_test.reshape(-1)

print(preds_1[:3])
print(ytest_1[:3])

diff = preds_1 - ytest_1
print(diff[:3])

diff_abs = np.abs(diff)
print(diff_abs[:3])

avg = np.mean(diff_abs)
print('오차 평균 :', avg)
print('오차 평균 : {}달러'.format(int(avg * 1000)))
sess.close()
