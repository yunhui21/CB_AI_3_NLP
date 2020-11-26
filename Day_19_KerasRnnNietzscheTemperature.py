# Day_19_KerasRnnNietzscheTemperature.py
# Day_18_03 copy
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import numpy as np
from sklearn import preprocessing

np.set_printoptions(linewidth=1000)

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


def show_sampling_2(long_text, seq_len, temperature):
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

    model.fit(x, y, epochs=1, batch_size=32, verbose=2)
    print(model.evaluate(x, y, verbose=0))  # ensor

    # ----------------------------------------------------- #

    write_novel_y(model, vocab, seq_len, y, temperature)

# 가중치 비율에 맞게 선택
# def weighted_pick(preds):
#     t = np.cumsum(preds)
#     return np.searchsorted((t, np.random.rand(1)[0] * t[-1]))
#     # return np.searchsorted((t, np.random.rand(1)[0]))

def temperature_pick(preds, temperature):
    dist = np.log(preds) / temperature
    dist = np.exp(dist)
    return dist / np.sum(dist)

def write_novel_y(model, vocab, seq_len, y, temperature):
    start = np.random.randint(0, len(y) - 1 - seq_len, 1)
    print(start)

    start = start[0]
    indices = y[start:start+seq_len]
    print(indices)

    for i in range(100):
        xx = np.zeros([1, seq_len, len(vocab)])
        for j, pos in enumerate(indices):
            print(j, pos)
            xx[0, j, pos] = 1
        print(xx)

        preds = model.predict(xx)
        print(preds.shape)  # (1, 31)

        p = np.argmax(preds, axis=1)
        # p = temperature_pick(preds, temperature)


        indices[:-1] = indices[1:]
        indices[-1] = p

        print(vocab[p], end='')
    print()




f = open('data/nietzsche.txt', 'r', encoding='utf-8')
nietzsch = f.read().lower()
f.close()

long_text =  nietzsch[:1000]


# show_sampling_1(long_text, seq_len=20)
show_sampling_2(long_text, seq_len=20, temperature=0.0)
# show_sampling_2(long_text, seq_len=20, temperature=0.2)
# show_sampling_2(long_text, seq_len=20, temperature=0.5)
# show_sampling_2(long_text, seq_len=20, temperature=1.0)
# show_sampling_2(long_text, seq_len=20, temperature=1.2)
