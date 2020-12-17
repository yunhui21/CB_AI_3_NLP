# Day_34_02_exam_1.py
import tensorflow as tf
import numpy as np

# 자격증 1번 문제
# x, y 데이터에 대해 모델을 구축하세요.

x = [0, 1, 2 ,3, 4, 5, 6]
y = [-3, -2, -1, 0, 1, 2, 3]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, activation=None))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
              loss=tf.keras.losses.mse)

model.fit(x, y, epochs=100, verbose=2)

preds = model.predict(x, verbose=0)
preds_arg = preds.reshape(-1)
print(preds_arg)
