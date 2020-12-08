# ------------------------------ #
# 등수: 13
# 캐글: 0.50121
# 피씨: 0.50121
# ------------------------------ #

# urbanchanllange_softmax_v1.py
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing

np.set_printoptions(linewidth=1000)


def get_data_train(file_path):
    data = pd.read_csv(file_path)
    data = [data.S1, data.C1, data.S2, data.C2,
            data.S3, data.C3, data.S4, data.C4,
            data.S5, data.C5, data.CLASS]
    data = np.int32(data)           # (12, 25010)
    data = np.transpose(data)       # (25010, 12)
    # print(data.shape)

    x = data[:, :-1]
    y = data[:, -1:]
    # print(x.shape, y.shape)
    return np.float32(x), np.float32(y)


def get_data_test(file_path):
    data = pd.read_csv(file_path)

    ids = data.Id.values
    data = [data.S1, data.C1, data.S2, data.C2,
            data.S3, data.C3, data.S4, data.C4,
            data.S5, data.C5]
    data = np.float32(data)         # (11, 1000000)
    data = np.transpose(data)       # (1000000, 11)
    # print(data.shape)

    return np.float32(data), np.float32(ids)


def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')
    print('Id,CLASS', file=f)
    for i in range(len(ids)):
        print('{},{}'.format(ids[i], preds[i]), file=f)
    f.close()


def model_poker(x_train, y_train, x_test):
    x_train = preprocessing.scale(x_train)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=100, verbose=2)

    # sample_submission save
    preds = model.predict(x_test)
    preds_arg = np.argmax(preds, axis=1)
    return preds_arg


x_train, y_train = get_data_train('data/train.csv')
x_test, ids = get_data_test('data/test.csv')
# print(x_train.shape, y_train.shape)       # (25010, 11) (25010, 1)
# print(x_test.shape, ids.shape)            # (1000000, 11) (1000000,)
preds = model_poker(x_train, y_train, x_test)
make_submission('outputs/submission_ShinJaeSub.csv', np.int32(ids), preds)