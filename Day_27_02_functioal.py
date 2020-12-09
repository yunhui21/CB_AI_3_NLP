# Day_27_02_functioal.py
import tensorflow as tf
import numpy as np
from sklearn import datasets
# 문제
# linnerud 데이터셋에 대해 mae로 결과를 예측하세요.
# data = datasets.load_linnerud(return_X_y=True)
# x, y = datasets.load_linnerud(return_X_y=True)
# print(x)
# print(y)

def and_functional_basic():
    x, y = datasets.load_linnerud(return_X_y=True)

    input  = tf.keras.layers.Input(shape=[3])
    # 1
    # DENSE  = tf.keras.layers.Dense(5, activation='relu')(input)
    #
    # weight  = tf.keras.layers.Dense(1, activation=None)(DENSE)
    # waist  = tf.keras.layers.Dense(1, activation=None)(DENSE)
    # pulse  = tf.keras.layers.Dense(1, activation=None)(DENSE)

    # 2
    weight  = tf.keras.layers.Dense(5, activation='relu')(input)
    weight  = tf.keras.layers.Dense(1, activation=None)(weight)

    waist  = tf.keras.layers.Dense(5, activation='relu')(input)
    waist  = tf.keras.layers.Dense(1, activation=None)(waist)

    pulse  = tf.keras.layers.Dense(5, activation='relu')(input)
    pulse  = tf.keras.layers.Dense(1, activation=None)(pulse)

    model = tf.keras.Model(input, [weight, waist, pulse])

    # concat = tf.keras.layers.concatenate([weight, waist, pulse], axis=1)
    # model = tf.keras.Model(input, concat)
    # weight, waist, pulse는 연관성이 있다.

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.mse,
                  metrics=['mae'])
    y1, y2, y3 = y[:, :1], y[:, 1:2], y[:,2:3]
    history = model.fit(x, [y1,y2,y3], epochs=1000, verbose=2)
    print('acc:', model.evaluate(x, [y1,y2,y3], verbose=0))
    # acc: [483.4224853515625, 433.32593, 4.404439, 45.69212, 16.087534, 1.5452023, 5.5819106]

    print(history.history.keys())
    # 에포크 출력 결과
    # dict_keys(['loss', 'dense_1_loss', 'dense_3_loss', 'dense_5_loss',
    # 'dense_1_mean_absolute_error', 'dense_3_mean_absolute_error',
    # 'dense_5_mean_absolute_error'])
and_functional_basic()



