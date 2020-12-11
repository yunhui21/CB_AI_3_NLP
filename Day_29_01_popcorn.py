# Day_29_01_popcorn.py
# 충북대학교 인공지능 강의
# https://support.google.com/chrome/answer/6130773?hl=ko
# 공업수학 http://www.kocw.or.kr/home/search/kemView.do?kemId=1350911
import tensorflow as tf
import numpy as np
import re
from sklearn import model_selection,feature_extraction, linear_model
import matplotlib.pyplot as plt
import gensim
import nltk
import pandas as pd

def tokenizing_and_padding(x, vocab_size, seq_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x) # 88581

    # print(len(tokenizer.index_word))     # 갯수만 세어보기

    # 숫자로 변환되기 전의 상태로 준비완료.
    x = tokenizer.texts_to_sequences(x) # text를 sequence 숫자로 변환

    # 250개가 80%, 500개는 90%
    # freqs = sorted([len(t) for t in x]) # 한번만 사용
    # plt.plot(freqs)
    # plt.show()

    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=seq_len)

    return x, tokenizer

# 문제
# 서브미션 파일을 만드세요.

def make_submission(idx, preds_bool, file_path):
    f = open(file_path, 'w', encoding='utf-8')

    f.write('"id", "sentiment"\n')
    for i, result in zip(ids, preds_bool):
        f.write('{},{}\n'.format(i, result))
    f.close()

def make_submission_for_deep(ids, x_test, model, tokenizer, seq_len, file_path):
    x_test = tokenizer.texts_to_sequences(x_test)  # text를 sequence 숫자로 변환
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=seq_len)

    preds = model.predict(x_test)
    preds_arg = preds.reshape(-1)
    preds_bool = np.int32(preds_arg > 0.5)

    make_submission(ids, preds_bool, 'popcorn_model/baseline.csv')

def model_baseline(x, y, ids, x_test):
    vocab_size, seq_len, n_feature = 2000, 200, 100
    x, tokenizer = tokenizing_and_padding(x, vocab_size, seq_len)

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data # submission 제출하는게 문제이므로 test는 없음
    # shuffle이 진행되면 순서가 바뀌어서 검증이
    # print(x.shape, y.shape)     # (1000, 200) (1000, 1)
    # ----------------------------------------------------------------------------- #
    # 문제
    # 학습하고 정확도를 구하는 나머지 코드를 만드세요.

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[seq_len]))
    model.add(tf.keras.layers.Embedding(vocab_size, n_feature))  # 입력(2차원), 출력(3차원)
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, batch_size=128, epochs = 10, verbose=2,
              validation_data=(x_valid, y_valid))

    #----------------------------------------------------------------#
    make_submission_for_deep(ids, x_test, model, tokenizer, seq_len, 'popcorn_model/baseline.csv')

# tfidf : Tern Frequencey-Inverse Document Frequency. 문서안에서 단어의 빈도수를 파악할때 사용하는 예전방법
def model_tfidf(x, y, ids, x_test):
    vocab_size, seq_len, n_feature = 2000, 200, 100
    x, tokenizer = tokenizing_and_padding(x, vocab_size, seq_len)
    # print(x.shape)      # (1000, 200)
    # 사용해서는 안되지만, tf-idf가 문자열 토큰을 필요로 하기 때문에.
    # tokenizing_and_padding 함수에서 texts_to_sequences 함수를 호출했기 때문에 성능이 좋지 않다.
    x = tokenizer.sequences_to_texts(x)

    tfidf = feature_extraction.text.TfidfVectorizer(
        min_df=0.0, analyzer='word', sublinear_tf=True,
        ngram_range=(1, 3), max_features=5000
    )
    x = tfidf.fit_transform(x)
    # print(x.shape)      # (1000, 5000)


    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data # submission 제출하는게 문제이므로 test는 없음
    # ------------------------------------------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight= 'balanced' )
    lr.fit(x_train, y_train)
    print('acc:', lr.score(x_valid, y_valid))

    #----------------------------------------------------------------#
    x_test = tokenizer.texts_to_sequences(x_test)  # text를 sequence 숫자로 변환
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=seq_len)

    x_test = tokenizer.sequences_to_texts(x_test)
    x_test = tfidf.fit_transform(x_test)

    preds = lr.predict(x_test)
    # print(preds.shape)
    # print(preds[:5])
    # preds_arg = preds.reshape(-1)
    # preds_bool = np.int32(preds_arg > 0.5)
    #
    make_submission(ids, preds, 'popcorn_model/tfidf.csv')


def model_word2vec(x, y, ids, x_test):
    vocab_size, seq_len, n_feature = 2000, 200, 100
    x, tokenizer = tokenizing_and_padding(x, vocab_size, seq_len)
    # print(x.shape)      # (1000, 200)
    # 사용해서는 안되지만, tf-idf가 문자열 토큰을 필요로 하기 때문에.
    # tokenizing_and_padding 함수에서 texts_to_sequences 함수를 호출했기 때문에 성능이 좋지 않다.
    x = tokenizer.sequences_to_texts(x)

    tfidf = feature_extraction.text.TfidfVectorizer(
        min_df=0.0, analyzer='word', sublinear_tf=True,
        ngram_range=(1, 3), max_features=5000
    )
    x = tfidf.fit_transform(x)
    # print(x.shape)      # (1000, 5000)


    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data # submission 제출하는게 문제이므로 test는 없음
    # ------------------------------------------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight= 'balanced' )
    lr.fit(x_train, y_train)
    print('acc:', lr.score(x_valid, y_valid))

    #----------------------------------------------------------------#
    x_test = tokenizer.texts_to_sequences(x_test)  # text를 sequence 숫자로 변환
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=seq_len)

    x_test = tokenizer.sequences_to_texts(x_test)
    x_test = tfidf.fit_transform(x_test)

    preds = lr.predict(x_test)
    # print(preds.shape)
    # print(preds[:5])
    # preds_arg = preds.reshape(-1)
    # preds_bool = np.int32(preds_arg > 0.5)
    #
    make_submission(ids, preds, 'popcorn_model/tfidf.csv')



popcorn = pd.read_csv('popcorn/labeledTrainData.tsv',
                      delimiter='\t', index_col=0)
# print(popcorn)

x = popcorn.review.values
y = popcorn.sentiment.values.reshape(-1, 1)
# print(x.dtype, y.dtype)     # object int64

# token의 길이의 분포를 파악한후

n_samples = 1000
x, y = x[:n_samples], y[:n_samples] # 에러나기전까지는 활성화 한다.
#
test_set = pd.read_csv('popcorn/testData.tsv',
                      delimiter='\t', index_col=0)

ids = test_set.index.values
x_test = test_set.review.values


# model_baseline(x, y, ids, x_test)
# model_tfidf(x, y, ids, x_test)
model_word2vec(x, y, ids, x_test)


# 1.baseline : loss: 0.1406 - acc: 0.9563 - val_loss: 0.9479 - val_acc: 0.6200
# 2.baseline_10 : loss: 0.0821 - acc: 0.9712 - val_loss: 0.4775 - val_acc: 0.8572



