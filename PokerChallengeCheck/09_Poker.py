# ------------------------------ #
# 등수: 9
# 캐글: 0.99701
# 피씨: 0.98892
# ------------------------------ #

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf


def poker_hand_model():
    poker_model = tf.keras.Sequential()
    poker_model.add(tf.keras.layers.Dense(256, activation='relu'))
    poker_model.add(tf.keras.layers.Dense(256, activation='relu'))
    poker_model.add(tf.keras.layers.Dense(256, activation='relu'))
    poker_model.add(tf.keras.layers.Dense(10, activation='softmax'))
    poker_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=tf.keras.losses.sparse_categorical_crossentropy,
                        metrics=['acc'])

    return poker_model


poker_hand_data_train = pd.read_csv('data/train.csv', index_col=0)
poker_hand_data_train = poker_hand_data_train.values
np.random.shuffle(poker_hand_data_train)

x_train = poker_hand_data_train[:, :-1]
y_train = poker_hand_data_train[:, -1:]
print(x_train.shape, y_train.shape)

poker_hand_data_test = pd.read_csv('data/test.csv', index_col=0)
x_test = poker_hand_data_test.values[:, :]
print(x_test.shape)

model = poker_hand_model()
hist = model.fit(x_train, y_train, epochs=128, batch_size=64, verbose=2,
                 validation_split=0.3)
predictions = model.predict(x_test)
print(predictions.shape)

classes = np.argmax(predictions, axis=1)
print(classes.shape)

poker_hand_data = pd.read_csv('data/test.csv')
pd.DataFrame({'Id': poker_hand_data.Id, 'CLASS': classes}).set_index('Id').to_csv('outputs/submission_castleshin.csv')