# Day_16_03_KerasRnnLstm.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

# 문제
# 예측한 결과에 대해 0.04보다 작다면 정담으로 인정해서
# 정확도가 얼마인지 알려주세요.
def get_data(seq_len=100, size=3000):
    np.random.seed(23)
    x, y = [], []
    for _ in range(size):
        values = np.random.rand(seq_len)    # 0~1까지의 값
        i0, i1 = np.random.choice(seq_len, 2, replace=False) # 중북된것없이.
        # print(values)

        # print(values)
        # print(np.min(values), np.max(values)) # 0.017533130833968458 0.9938808039095165
        # print(i0, i1)
        assert i0 != i1

        two_hots = np.zeros(seq_len)
        two_hots[[i0, i1]] = 1
        # print(two_hots)

        xx = np.transpose([values, two_hots])
        yy = values[i0]*values[i1]
        # print(xx.shape)

        x.append(xx)
        y.append(yy)

    x = np.float32(x)
    y = np.float32(y)

    return model_selection.train_test_split(x, y, train_size=0.8)

def simple_rnn():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[100,2]))

    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(30, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    return model

def lstm():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[100,2]))
    model.add(tf.keras.layers.LSTM(30, return_sequences=True))
    model.add(tf.keras.layers.LSTM(30, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    return model

def gru():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.GRU(30, return_sequences=True, input_shape=[100, 2]))
    model.add(tf.keras.layers.Input(shape=[100,2]))
    model.add(tf.keras.layers.GRU(30, return_sequences=True))
    model.add(tf.keras.layers.GRU(30, return_sequences=False))
    model.add(tf.keras.layers.Dense(1))

    return model
def show_history(history):
    # 문제
    # history 객체에 들어 있는 값으로 그래프를 그려주세요.
    plt.plot(history.history['loss'], 'r--', label= 'loss')
    plt.plot(history.history['val_loss'], 'g--', label= 'val_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.title('RNN')
    plt.show()

def show_model(model_func):

    x_train, x_test, y_train, y_test = get_data(seq_len=100, size=3000)

    # print(x_train.shape, x_test.shape) # (2400, 100, 2) (600, 100, 2)
    # print(y_train.shape, y_test.shape) # (2400,) (600,)

    model = model_func
    # model.summary()
    # return
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mse)

    history = model.fit(x_train, y_train, epochs=30, verbose=2, batch_size=32, validation_split=0.2)
    print(history)
    print(history.history)
    # {'loss': [0.059102512896060944, 0.05276212841272354, 0.052386924624443054],
    #  'val_loss': [0.052296195179224014, 0.05189915746450424, 0.052378978580236435]}


    # preds = model.predict(x_test)
    # print(preds.shape) # (600, 100, 1)

    # preds = preds.reshape(-1)
    # print('acc:', np.mean(np.abs(preds - y_test) <= 0.04))
    show_history(history)


# show_model(simple_rnn)
# show_model(lstm)
show_model(gru())

# simple_rnn
# Epoch 30/30
# 60/60 - 1s - loss: 0.0501 - val_loss: 0.0506
# acc: 0.11666666666666667
# Total params: 2,851
# Trainable params: 2,851
# Non-trainable params: 0



# lstm()
# Epoch 30/30
# 60/60 - 2s - loss: 0.0531 - val_loss: 0.0535
# acc: 0.09833333333333333
# Total params: 11,311
# Trainable params: 11,311
# Non-trainable params: 0


# gru()
# Epoch 30/30
# 60/60 - 2s - loss: 0.0011 - val_loss: 7.7690e-04
# acc: 0.9183333333333333
# Total params: 8,671
# Trainable params: 8,671
# Non-trainable params: 0
