# Day_15_02_KerasMultiLayers.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

# 문제
# mnist 데이터에 대해 소프트맥스 리그레션으로 검사 데이터에 대해 정확도를 구하세요.


data = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = data

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# 784->256->256->10
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2,
          validation_data=(x_test, y_test))