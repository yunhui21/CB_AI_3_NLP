# Day_18_03_KerasRnnNietzsche.py
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

# 문제
# Day_12_01_RnnBasic_final.py 파일의 코드를 케라스 RNN 버전으로 수정하세요
# (long_text 변수는 니체 파일로 대체합니다)
# x: tenso
# y: ensor

# 문제
# 주식 시계열 데이터처럼 일정 갯수의 문자를 이용해서 그 다음 글자를 예측하는 모델로 수정하세요
# x: ten ens nso
# y: s   o   r


def make_data_1(long_text, seq_len):
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    words = [long_text[i:i+seq_len+1] for i in range(len(long_text)-seq_len)]

    x, y = [], []
    for word in words:
        onehot = lb.transform(list(word))

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))

    return np.float32(x), np.int32(y), lb


def show_sampling_1(long_text, seq_len):
    x, y, lb = make_data_1(long_text, seq_len)
    vocab = lb.classes_

    print(x.shape, y.shape)     # (980, 20, 31) (980, 20)

    _, seq_len, n_classes = x.shape
    hidden_size = 21

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[seq_len, n_classes]),
        tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=10, batch_size=32, verbose=2)
    # print(model.evaluate(x, y, verbose=0))

    preds = model.predict(x)
    preds_arg = np.argmax(preds, axis=2)    # (1, 5)

    print('*' + ''.join(vocab[preds_arg[0]]), end='')

    for arg in preds_arg[1:]:
        print(''.join(vocab[arg[-1]]), end='')
    print()


# ------------------------------------ #


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


def show_sampling_2(long_text, seq_len):
    x, y, lb = make_data_2(long_text, seq_len)
    vocab = lb.classes_

    print(x.shape, y.shape)     # (980, 20, 31) (980,)

    _, seq_len, n_classes = x.shape
    hidden_size = 21

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[seq_len, n_classes]),
        tf.keras.layers.SimpleRNN(hidden_size, return_sequences=False),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=10, batch_size=32, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    preds = model.predict(x)
    print(preds.shape)                      # (980, 31)

    preds_arg = np.argmax(preds, axis=1)    # (980,)

    print(''.join(vocab[preds_arg]))


f = open('data/nietzsche.txt', 'r', encoding='utf-8')
nietzsche = f.read().lower()
f.close()

long_text = nietzsche[:1000]

# show_sampling_1(long_text, seq_len=20)
show_sampling_2(long_text, seq_len=20)
