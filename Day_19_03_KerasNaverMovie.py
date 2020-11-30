# Day_19_03_KerasNaverMovie.py
import csv
import re
import nltk
import collections
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 문제
# Day_07_02_NaverMovie.py 파일을 케라스 모델로 수정하세요

# 문제
# 원핫 버전을 복사해서 입베딩 레이어 버전으로 수정하세요.

# 문제
# 우리가 작성한 리뷰에 대해 긍정 또는 부정을 판단하는 함수를 만드세요.(원핫 버전)

def get_data(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    f.readline()

    x, y = [], []
    for _, doc, label in csv.reader(f, delimiter='\t'):
        x.append(clean_str(doc).split())
        y.append(label)

    f.close()

    # return x, y
    return x[:1000], y[:1000]


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip() if TREC else string.strip().lower()


def show_freq_dist(documents):
    print(len(documents))
    print(documents[0])
    print(max([len(doc) for doc in documents]))

    heights = [len(doc) for doc in documents]
    heights = sorted(heights)

    plt.plot(range(len(heights)), heights)
    plt.show()


def show_prediction(new_review, model, tokenizer, max_len, one_hot):
    pass

def rnn_model_onehost():
    pass

def rnn_model_embedding():
    x_train, y_train = get_data('data/ratings_train.txt')
    x_test, y_test = get_data('data/ratings_test.txt')

    print(y_train[:5])                  # ['0', '1', '0', '0', '1']

    y_train = np.int32(y_train).reshape(-1, 1)
    y_test = np.int32(y_test).reshape(-1, 1)

    print(x_train[0])                   # ['아', '더빙', '진짜', '짜증나네요', '목소리']
    print(len(x_train), len(x_test))    # 1000 1000

    # show_freq_dist(x_train)           # 토큰 40개 정도면 대부분의 리뷰를 포함

    vocab_size = 200
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    print(x_train[0])                   # [22, 362, 6, 826, 827]

    max_len = 25
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
    print(x_train[0])                   # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 22 6]
    print(x_train.shape, x_test.shape)  # (1000, 25) (1000, 25)

    # 피처 생성
    onehot = np.eye(vocab_size, dtype=np.int32)

    x_train = onehot[x_train]
    x_test = onehot[x_test]
    print(x_train.shape, x_test.shape)  # (1000, 25, 200) (1000, 25, 200)

    # --------------------------- #

    hidden_size = 21

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[max_len]),
        tf.keras.layers.Embedding(vocab_size, 100),
        tf.keras.layers.LSTM(hidden_size, return_sequences=False),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))


# 1. 필요없는 코드 삭제
# 2. word2vec으로 변환
# 3. 나의 리뷰 예측

rnn_model_embedding()