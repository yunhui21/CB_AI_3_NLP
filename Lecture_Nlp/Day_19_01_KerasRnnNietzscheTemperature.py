# Day_19_01_KerasRnnNietzscheTemperature.py
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# Day_18_03_KerasRnnNietzsche.py 파일을 복사해서 사용합니다


def make_data_2(long_text, seq_len):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    words = [long_text[i:i+seq_len+1] for i in range(len(long_text)-seq_len)]

    x, y = [], []
    for word in words:
        onehot = lb.transform(list(word))

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[-1]))                # 수정한 부분

    return np.float32(x), np.int32(y), lb


def show_sampling_2(long_text, seq_len, temperature):
    x, y, lb = make_data_2(long_text, seq_len)
    vocab = lb.classes_

    # print(x.shape, y.shape)     # (980, 20, 31) (980,)

    _, seq_len, n_classes = x.shape
    hidden_size = 21

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[seq_len, n_classes]),
        tf.keras.layers.LSTM(hidden_size, return_sequences=False),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, batch_size=32, verbose=2)
    # print(model.evaluate(x, y, verbose=0))

    # ---------------------------------- #

    # 1번 케라스 창시자 코드
    # start = np.random.randint(0, len(y) - 1 - seq_len, 1)
    # # print(start)        # [760]
    #
    # start = start[0]
    # yy = y[start:start+seq_len]
    # # print(yy)           # [ 8 25  1 22 23 12 24 12 20 25  1 12 27 12 23 29  1 17 16 20]
    #
    # if len(temperature) <= 0:
    #     write_novel_y(model, vocab, seq_len, yy, 0.0)          # weighted pick
    # else:
    #     for t in temperature:                               # temperature pick
    #         print(t, '-' * 30)
    #         write_novel_y(model, vocab, seq_len, yy, t)

    # 문제
    # x 데이터를 이용해서 소설을 쓰는 코드를 구현하세요

    # 2번 우리가 만든 코드
    start = np.random.randint(0, len(y) - 1 - seq_len, 1)
    start = start[0]

    xx = x[start]
    xx = xx[np.newaxis]
    # print(x.shape, xx.shape)          # (980, 20, 31) (1, 20, 31)

    if len(temperature) <= 0:
        write_novel_x(model, vocab, xx, 0.0)          # weighted pick
    else:
        for t in temperature:                         # temperature pick
            print(t, '-' * 30)
            write_novel_x(model, vocab, xx, t)


# 가중치 비율에 맞게 선택
# def weighted_pick(preds):
#     t = np.cumsum(preds)
#     return np.searchsorted((t, np.random.rand(1)[0] * t[-1]))
#     # return np.searchsorted((t, np.random.rand(1)[0]))


def temperature_pick(preds, temperature):
    if temperature > 0:
        preds = np.log(preds) / temperature
        preds = np.exp(preds)
        preds = preds / np.sum(preds)

    # weighted_pick 함수
    t = np.cumsum(preds)
    # return np.searchsorted(t, np.random.rand(1)[0] * t[-1])
    return np.searchsorted(t, np.random.rand(1)[0])


# 케라스 창시자가 만든 코드
def write_novel_y(model, vocab, seq_len, yy, temperature):
    for i in range(100):
        xx = np.zeros([1, seq_len, len(vocab)], dtype=np.int32)
        for j, pos in enumerate(yy):
            # print(j, pos)
            xx[0, j, pos] = 1

        # print(xx)

        preds = model.predict(xx)
        # print(preds.shape)          # (1, 31)

        # p = np.argmax(preds, axis=1)
        # p = p[0]
        p = temperature_pick(preds, temperature)
        # print(p, vocab[p])

        # p = 71
        # [ 8 25  1 22 23]
        # [25  1 22 23 71]
        yy[:-1] = yy[1:]
        yy[-1] = p
        # print(yy)

        print(vocab[p], end='')
    print()


# 우리가 만든 코드
def write_novel_x(model, vocab, xx, temperature):
    for i in range(100):
        preds = model.predict(xx)
        # print(preds.shape)          # (1, 31)

        p = temperature_pick(preds, temperature)

        # xx[:, :-1, :] = xx[:, 1:, :]       # xx[:-1] = xx[1:]
        xx[0, :-1] = xx[0, 1:]

        # print(xx[0, -1])

        xx[0, -1] = 0
        xx[:, -1, p] = 1                    # xx[-1] = p
        # print(xx[0, -1])

        print(vocab[p], end='')
    print()


f = open('data/nietzsche.txt', 'r', encoding='utf-8')
nietzsche = f.read().lower()
f.close()

long_text = nietzsche[:1000]

# show_sampling_2(long_text, seq_len=20, temperature=[])                      # weighted pick
show_sampling_2(long_text, seq_len=20, temperature=[0.2, 0.5, 1.0, 1.2])    # temperature pick
