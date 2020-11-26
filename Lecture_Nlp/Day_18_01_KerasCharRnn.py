# Day_18_01_KerasCharRnn.py
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

# RnnBasic에서 수업했던 내용을 케라스 버전으로 변환

# 문제
# 'tensor'를 x, y로 변환해서 케라스 모델로 예측하세요 (소프트맥스 리그레션)
# x: tenso
# y: ensor

# 문제
# 소프트맥스 리그레션으로 만든 모델을 케라스 RNN 버전으로 수정하세요 (dense 버전)
# (SimpleRNN 레이어 추가하세요)

# 문제
# dense 버전으로 만든 코드를 sparse 버전으로 수정하세요
# (categorical_crossentropy를 sparse_categorical_crossentropy 함수로 바꾸세요)


def char_rnn_softmax():
    word = 'tensor'

    enc = preprocessing.LabelBinarizer()
    onehot = enc.fit_transform(list(word))

    print(enc.classes_)     # ['e' 'n' 'o' 'r' 's' 't'] vocab
    print(onehot)

    x, y = onehot[:-1], onehot[1:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[x.shape[-1]]),
        tf.keras.layers.Dense(len(enc.classes_), activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    preds = model.predict(x)
    print(preds.shape)                          # (5, 6)

    preds_arg = np.argmax(preds, axis=1)
    print(''.join(enc.classes_[preds_arg]))     # ensor


def char_rnn_simple_rnn_dense():
    word = 'tensor'

    enc = preprocessing.LabelBinarizer()
    onehot = enc.fit_transform(list(word))

    x, y = onehot[:-1], onehot[1:]

    # rnn은 3차원 입력을 사용하니까
    x = x[np.newaxis]
    y = y[np.newaxis]
    print(x.shape, y.shape)     # (1, 5, 6) (1, 5, 6)

    model = tf.keras.Sequential([
        # tf.keras.layers.Input(shape=[x.shape[1], x.shape[2]]),    # seq_len, n_features
        tf.keras.layers.Input(shape=[5, 6]),
        tf.keras.layers.SimpleRNN(21, return_sequences=True),
        tf.keras.layers.Dense(len(enc.classes_), activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    preds = model.predict(x)
    print(preds.shape)                          # (1, 5, 6)

    preds_arg = np.argmax(preds, axis=2)
    print(preds_arg.shape)                      # (1, 5)

    print(''.join(enc.classes_[preds_arg[0]]))  # ensor


def char_rnn_simple_rnn_sparse():
    word = 'tensor'

    enc = preprocessing.LabelBinarizer()
    onehot = enc.fit_transform(list(word))

    x, y = onehot[:-1], onehot[1:]
    y = np.argmax(y, axis=1)        # sparse

    # rnn은 3차원 입력을 사용하니까
    x = x[np.newaxis]
    y = y[np.newaxis]
    print(x.shape, y.shape)         # (1, 5, 6) (1, 5)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[x.shape[1], x.shape[2]]),    # seq_len, n_features
        tf.keras.layers.SimpleRNN(21, return_sequences=True),
        tf.keras.layers.Dense(len(enc.classes_), activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,     # sparse
                  metrics=['acc'])

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    # sparse 버전과 dense 버전은 아래 코드에 대해서는 차이가 없다 
    preds = model.predict(x)
    print(preds.shape)                          # (1, 5, 6)

    preds_arg = np.argmax(preds, axis=2)
    print(preds_arg.shape)                      # (1, 5)

    print(''.join(enc.classes_[preds_arg[0]]))  # ensor


# char_rnn_softmax()
# char_rnn_simple_rnn_dense()
char_rnn_simple_rnn_sparse()

