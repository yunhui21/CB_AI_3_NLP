# Day_16_03_KerasRnnLstm.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

# 문제
# 예측한 결과에 대해 0.04보다 작다면 정답으로 인정해서
# 정확도가 얼마인지 알려주세요


def get_data(seq_len=100, size=3000):
    np.random.seed(23)
    x, y = [], []
    for _ in range(size):
        values = np.random.rand(seq_len)
        i0, i1 = np.random.choice(seq_len, 2, replace=False)

        # print(values)
        # print(np.min(values), np.max(values))   # 0.01398 0.99604
        # print(i0, i1)
        assert i0 != i1

        two_hots = np.zeros(seq_len)
        two_hots[[i0, i1]] = 1
        # print(two_hots)

        xx = np.transpose([values, two_hots])
        yy = values[i0] * values[i1]
        # print(xx.shape)

        x.append(xx)
        y.append(yy)

    x = np.float32(x)
    y = np.float32(y)

    return model_selection.train_test_split(x, y, train_size=0.8)


def simple_rnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[100, 2]))
    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    return model


def lstm():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[100, 2]))
    model.add(tf.keras.layers.LSTM(30, return_sequences=True))
    model.add(tf.keras.layers.LSTM(30, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    return model


def gru():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.GRU(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.Input(shape=[100, 2]))
    model.add(tf.keras.layers.GRU(30, return_sequences=True))
    model.add(tf.keras.layers.GRU(30, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    return model


def show_model(model_func):
    x_train, x_test, y_train, y_test = get_data(seq_len=100, size=3000)
    # print(x_train.shape, x_test.shape)  # (2400, 100, 2) (600, 100, 2)
    # print(y_train.shape, y_test.shape)  # (2400,) (600,)

    model = model_func()
    model.summary()
    return

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)

    model.fit(x_train, y_train, epochs=30, verbose=2, batch_size=32, validation_split=0.2)

    preds = model.predict(x_test)
    # print(preds.shape)

    preds = preds.reshape(-1)
    print('acc :', np.mean(np.abs(preds - y_test) <= 0.04))


show_model(simple_rnn)
# show_model(lstm)
# show_model(gru)

# simple_rnn
# Epoch 30/30
# 60/60 - 2s - loss: 0.0484 - val_loss: 0.0500
# acc : 0.11333333333333333

# lstm
# Epoch 30/30
# 60/60 - 2s - loss: 0.0041 - val_loss: 0.0036
# acc : 0.6333333333333333

# gru
# Epoch 30/30
# 60/60 - 3s - loss: 7.7123e-04 - val_loss: 9.3915e-04
# acc : 0.84

# history + 그래프 표시
