# Day_29_01_popcorn.py
import tensorflow as tf
import numpy as np
import re
from sklearn import model_selection, feature_extraction, linear_model
import matplotlib.pyplot as plt
import gensim
import nltk
import pandas as pd


def tokenizing_and_padding(x, vocab_size, seq_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x)

    # print(len(tokenizer.index_word))      # 88582
    # print(x[1])   # \The Classic War of the Worlds\" by
    # print(tokenizer.index_word[1])

    x = tokenizer.texts_to_sequences(x)
    # print(x[1])   # [1, 372, 276, 4, 1, 30, 6, 2, ...]

    # 250개는 80%, 500개는 90% 포함
    # freqs = sorted([len(t) for t in x])
    # plt.plot(freqs)
    # plt.show()

    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=seq_len)

    return x, tokenizer


# 문제
# 서브미션 파일을 만드세요
def make_submission(ids, preds_bool, file_path):
    f = open(file_path, 'w', encoding='utf-8')

    f.write('"id","sentiment"\n')
    for i, result in zip(ids, preds_bool):
        f.write('{},{}\n'.format(i, result))

    f.close()


# 사용 모델: baseline, rnn, cnn, cnn-tf.data
def make_submission_for_deep(ids, x_test, model, tokenizer, seq_len, file_path):
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=seq_len)

    preds = model.predict(x_test)
    preds_arg = preds.reshape(-1)
    preds_bool = np.int32(preds_arg > 0.5)

    make_submission(ids, preds_bool, file_path)


# 사용 모델: word2vec, word2vec_nltk
def make_submission_for_word2vec(ids, x_test, model, word2vec, n_features, idx2word, file_path):
    x_test = [s.lower().split() for s in x_test]

    features = [make_features_for_word2vec(
        tokens, word2vec, n_features, idx2word) for tokens in x_test]
    x_test = np.vstack(features)

    preds = model.predict(x_test)
    make_submission(ids, preds, file_path)


# 사용 모델: word2vec, word2vec_nltk
# tokens: [the, in, happy, in]
def make_features_for_word2vec(tokens, word2vec, n_features, idx2word):
    binds, n_words = np.zeros(n_features), 0

    for w in tokens:
        if w in idx2word:
            binds += word2vec.wv[w]
            n_words += 1

    return binds / (n_words if n_words > 0 else 1)


def model_baseline(x, y, ids, x_test):
    vocab_size, seq_len, n_features = 2000, 200, 100
    x, tokenizer = tokenizing_and_padding(x, vocab_size, seq_len)

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data

    # ------------------------------------- #

    # 문제
    # 학습하고 정확도를 구하는 나머지 코드를 만드세요
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[seq_len]))
    model.add(tf.keras.layers.Embedding(vocab_size, n_features))
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train,
              epochs=10, batch_size=128, verbose=2,
              validation_data=(x_valid, y_valid))

    # ----------------------------------------- #

    make_submission_for_deep(ids, x_test, model, tokenizer, seq_len,
                             'popcorn_model/baseline.csv')


# tfidf: Term Frequency-Inverse Document Frequency
def model_tfidf(x, y, ids, x_test):
    vocab_size, seq_len, n_features = 2000, 200, 100
    x, tokenizer = tokenizing_and_padding(x, vocab_size, seq_len)
    # print(x.shape)      # (1000, 200)

    # 사용해서는 안되지만, tf-idf가 문자열 토큰을 필요로 하기 때문에.
    # tokenizing_and_padding 함수에서 texts_to_sequences 함수를 호출했기 때문에 성능이 좋지 않다
    x = tokenizer.sequences_to_texts(x)

    tfidf = feature_extraction.text.TfidfVectorizer(
        min_df=0.0, analyzer='word', sublinear_tf=True,
        ngram_range=(1, 3), max_features=5000
    )
    x = tfidf.fit_transform(x)
    # print(x.shape)      # (1000, 5000)

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data

    # ------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight='balanced')
    lr.fit(x_train, y_train)
    print('acc :', lr.score(x_valid, y_valid))

    # ----------------------------------------- #

    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=seq_len)

    x_test = tokenizer.sequences_to_texts(x_test)
    x_test = tfidf.fit_transform(x_test)

    preds = lr.predict(x_test)
    # print(preds.shape)        # (25000,)
    # print(preds[:5])          # [1 1 1 0 1]

    make_submission(ids, preds, 'popcorn_model/tfidf.csv')


# gensim
def model_word2vec(x, y, ids, x_test):
    x = [s.lower().split() for s in x]

    n_features = 100
    word2vec = gensim.models.Word2Vec(
        x, workers=4, size=n_features,
        min_count=40, window=10, sample=0.001
    )
    # print(word2vec)   # Word2Vec(vocab=9207, size=100, alpha=0.025)

    # list -> set: 검색 성능 향상
    idx2word = set(word2vec.wv.index2word)

    features = [make_features_for_word2vec(
        tokens, word2vec, n_features, idx2word) for tokens in x]
    x = np.vstack(features)
    # print(x.shape)            # (1000, 100)

    # review 1개: [the, in, happy, in]
    # the  : 1 0 0 0 0
    # in   : 0 0 1 0 0
    # happy: 0 0 0 1 0
    # in   : 0 0 1 0 0
    # ----------------
    #        1 0 2 1 0 / 4

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data

    # ------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight='balanced')
    lr.fit(x_train, y_train)
    print('acc :', lr.score(x_valid, y_valid))

    # ----------------------------------------- #

    # 문제
    # 위의 코드를 사용해서 서브미션 파일을 만드세요
    make_submission_for_word2vec(
        ids, x_test, lr, word2vec, n_features, idx2word,
        'popcorn_model/word2vec.csv'
    )


def model_word2vec_nltk(x, y, ids, x_test):
    # x = [s.lower().split() for s in x]

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    sents = [tokenizer.tokenize(s.lower()) for s in x]
    # print(sents[1])   # ['the', 'classic', 'war', 'of', ...]

    st = nltk.stem.PorterStemmer()
    sents_stem = [[st.stem(w) for w in s] for s in sents]

    # 문제
    # 불용어를 제거하세요
    stop_words = nltk.corpus.stopwords.words('english')
    sents_token = [[w for w in s if w not in stop_words] for s in sents_stem]
    x = sents_token
    # print(sents_token[1])     # ['classic', 'war', 'world', 'timothi', ...]

    # ---------------------------- #
    # 아래 코드는 model_word2vec 함수와 100% 동일

    n_features = 100
    word2vec = gensim.models.Word2Vec(
        x, workers=4, size=n_features,
        min_count=40, window=10, sample=0.001
    )
    # print(word2vec)   # Word2Vec(vocab=113, size=100, alpha=0.025)

    # list -> set: 검색 성능 향상
    idx2word = set(word2vec.wv.index2word)

    features = [make_features_for_word2vec(
        tokens, word2vec, n_features, idx2word) for tokens in x]
    x = np.vstack(features)
    # print(x.shape)            # (1000, 100)

    # review 1개: [the, in, happy, in]
    # the  : 1 0 0 0 0
    # in   : 0 0 1 0 0
    # happy: 0 0 0 1 0
    # in   : 0 0 1 0 0
    # ----------------
    #        1 0 2 1 0 / 4

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data

    # ------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight='balanced')
    lr.fit(x_train, y_train)
    print('acc :', lr.score(x_valid, y_valid))

    # ----------------------------------------- #

    # 학습 데이터에 대해서는 stemming, 불용어 처리 등을 적용했기 때문에
    # 검사 데이터와 토큰에 있어서 많이 다름에도 불구하고 비슷한 결과를 보여줌
    # acc : 0.8544
    # make_submission_for_word2vec(
    #     ids, x_test, lr, word2vec, n_features, idx2word,
    #     'popcorn_model/word2vec_nltk.csv'
    # )

    # 문제
    # 앞에서 사용한 전처리 코드를 사용해서 서브미션 파일을 만드세요
    sents = [tokenizer.tokenize(s.lower()) for s in x_test]
    sents_stem = [[st.stem(w) for w in s] for s in sents]
    sents_token = [[w for w in s if w not in stop_words] for s in sents_stem]
    x_test = sents_token

    preds = lr.predict(x_test)
    make_submission(ids, preds, 'popcorn_model/word2vec_nltk.csv')


popcorn = pd.read_csv('popcorn/labeledTrainData.tsv',
                      delimiter='\t',
                      index_col=0)
# print(popcorn)

x = popcorn.review.values
y = popcorn.sentiment.values.reshape(-1, 1)
# print(x.dtype, y.dtype)         # object int64

n_samples = 1000
# x, y = x[:n_samples], y[:n_samples]

test_set = pd.read_csv('popcorn/testData.tsv',
                       delimiter='\t',
                       index_col=0)

ids = test_set.index.values
x_test = test_set.review.values

# model_baseline(x, y, ids, x_test)
# model_tfidf(x, y, ids, x_test)
# model_word2vec(x, y, ids, x_test)
model_word2vec_nltk(x, y, ids, x_test)


# baseline: 0.8596
# tfidf   : 0.8742
# word2vec: 0.8228
# nltk    : 0.8544

