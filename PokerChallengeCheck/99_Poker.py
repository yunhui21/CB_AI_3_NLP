# 99_Poker.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection

poker = pd.read_csv('data/train.csv', index_col=0)
# print(poker)

x = poker.values[:, :-1]
y = poker.values[:, -1:]

x = np.float32(x)

# 스케일링은 엄청난 효과 발생
# x = preprocessing.scale(x)

# 원핫 벡터 변환하면 엄청난 효과 발생
x_onehot = []
for i in range(x.shape[1]):
    xx = preprocessing.LabelBinarizer().fit_transform(x[:, i])
    x_onehot.append(xx)

x = np.hstack(x_onehot)

# ----------------------------------- #

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.8)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train,
          epochs=100, batch_size=100, verbose=2,
          validation_data=(x_valid, y_valid))
