# Day_18_03_KerasRnnNieche.py

# Day_18_01_KerasCharRnn.py
# Day_12_01_RnnBasic_final.py
# Day_11_2_RnnBasic_4_different.py
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)


# 문제
# 엄청나게 긴 문장에 대해 동작하는 버전을 만드세요.

# 문제
# 주식 시계열 데이터처럼 일정 개숫의 문자를 이용해서 그 다음 글자를 예측하는
def make_data_1(long_text, seq_len):
    e1 = preprocessing.LabelBinarizer()
    e1.fit(list(long_text))

    words = [long_text[i:i + seq_len + 1] for i in range(len(long_text) - seq_len)]

    x, y = [], []
    for word in words:  # list형태로 들어오기만 하면 된다.
        onehot = e1.transform(list(word))  # abcd, efgh

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[1:], axis=1))
    # exit(-1) # ['c' 'e' 'f' 'l' 'n' 'o' 'r' 's' 't' 'w' 'y']

    return np.float32(x), np.int32(y), e1


def show_sampling_1(long_text, seq_len):
    x, y, e1 = make_data_1(long_text, seq_len)
    vocab = e1.classes_

    print(x.shape, y.shape)

    _, seq_len, n_classes = x.shape
    hidden_size = 21


    model = tf.keras.Sequential(
        [
            # tf.keras.layers.Input(shape=[x.shape[1],x.shape[2]]),
            tf.keras.layers.Input(shape=[seq_len, n_classes]),
            tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ]
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, batch_size=32, verbose=2)
    print(model.evaluate(x, y, verbose=0))  # ensor

    preds = model.predict(x)  # (1, 5, 6)
    preds_arg = np.argmax(preds, axis=2)  # (1, 5)

    print('*' + ''.join(vocab[preds_arg[0]]), end='')

    for arg in preds_arg[1:]:
        print(''.join(vocab[arg[-1]]), end='')
    print()

#--------------------------------------------#

def make_data_2(long_text, seq_len):
    e1 = preprocessing.LabelBinarizer()
    e1.fit(list(long_text))

    words = [long_text[i:i + seq_len + 1] for i in range(len(long_text) - seq_len)]

    x, y = [], []
    for word in words:  # list형태로 들어오기만 하면 된다.
        onehot = e1.transform(list(word))  # abcd, efgh

        x.append(np.float32(onehot[:-1]))
        y.append(np.argmax(onehot[-1]))
    # exit(-1) # ['c' 'e' 'f' 'l' 'n' 'o' 'r' 's' 't' 'w' 'y']

    return np.float32(x), np.int32(y), e1


def show_sampling_2(long_text, seq_len):
    x, y, e1 = make_data_2(long_text, seq_len)
    vocab = e1.classes_

    print(x.shape, y.shape) # (980, 20, 31) (980,)

    _, seq_len, n_classes = x.shape
    hidden_size = 21


    model = tf.keras.Sequential(
        [
            # tf.keras.layers.Input(shape=[x.shape[1],x.shape[2]]),
            tf.keras.layers.Input(shape=[seq_len, n_classes]),
            tf.keras.layers.SimpleRNN(hidden_size, return_sequences=False),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ]
    )
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=100, batch_size=32, verbose=2)
    print(model.evaluate(x, y, verbose=0))  # ensor

    preds = model.predict(x)  # (1, 5, 6)
    preds_arg = np.argmax(preds, axis=1)  # (1, 5)

    print('*' + ''.join(vocab[preds_arg]), end='')


f = open('data/nietzsche.txt', 'r', encoding='utf-8')
nietzsch = f.read().lower()
f.close()

long_text =  nietzsch[:1000]


# show_sampling_1(long_text, seq_len=20)
show_sampling_2(long_text, seq_len=20)
