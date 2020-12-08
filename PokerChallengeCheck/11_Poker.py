# ------------------------------ #
# 등수: 11
# 캐글: 0.99212
# 피씨: 0.99265
# ------------------------------ #

import numpy as np
import pandas as pd
import tensorflow as tf


def make_train_data(dir):
    pi = pd.read_csv(dir)
    pi = pi.drop(['Id', 'S1', 'S2', 'S3', 'S4', 'S5'], axis=1)

    y = pd.get_dummies(pi['CLASS'])
    x = pi.values[:, :-1]
    x = np.sort(x, axis=1)

    return x, y


def make_test_data(dir):
    pi = pd.read_csv(dir)

    pi = pi.drop(['Id', 'S1', 'S2', 'S3', 'S4', 'S5'], axis=1)
    x = pi.values
    x = np.sort(x, axis=1)

    return x


def models():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(9, activation='softmax'))

    return model


def run(model_function, x, y, test_x):
    model = model_function()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])
    model.fit(x, y, epochs=10, verbose=2, validation_split=0.2)
    print(model.evaluate(x, y, verbose=2))
    preds = model.predict(test_x)
    preds_arg = np.argmax(preds, axis=1)

    return preds_arg


x, y = make_train_data("data/train.csv")
test_x = make_test_data("data/test.csv")

answer = run(models, x, y, test_x)

submit = pd.read_csv("data/sample_submission.csv")
# submit = submit.drop(['C1', 'S1', 'C2', 'S2', 'C3', 'S3', 'C4', 'S4', 'C5', 'S5'], axis = 1)
submit['CLASS'] = answer
submit.to_csv("outputs/submission_YUlee.csv", index=False)
