# Day_27_02_functioal.py
import tensorflow as tf
import numpy as np
from sklearn import datasets
# 문제
# linnerud 데이터셋에 대해 mae로 결과를 예측하세요.
# x, y = datasets.load_linnerud(return_X_y=True)
# print(x)
# print(y)

def and_functional_basic():
    x, y = datasets.load_linnerud(return_X_y=True)

    input  = tf.keras.layers.Input(shape=[3])

    weight  = tf.keras.layers.Dense(5, activation='relu')(input)
    weight  = tf.keras.layers.Dense(1, activation=None)(weight)

    waist  = tf.keras.layers.Dense(5, activation='relu')(input)
    waist  = tf.keras.layers.Dense(1, activation=None)(waist)

    pulse  = tf.keras.layers.Dense(5, activation='relu')(input)
    pulse  = tf.keras.layers.Dense(1, activation=None)(pulse)

    # tf.keras
    model = tf.keras.Model(input, [weight, waist, pulse])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.mse,
                  metrics=['mae'])
    y1, y2, y3 = y[:, :1], y[:, 1:2], y[:,2:3]
    model.fit(x, [y1,y2,y3], epochs=1000, verbose=2)
    print('acc:', model.evaluate(x, [y1,y2,y3], verbose=0))
    # acc: [483.4224853515625, 433.32593, 4.404439, 45.69212, 16.087534, 1.5452023, 5.5819106]
and_functional_basic()



