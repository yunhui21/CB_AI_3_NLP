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

# nltk.download('stopwords')
# nltk.download('gutenberg')
# nltk.download('webtext')
# nltk.download('reuters')

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

    make_submission(ids, preds_bool, file_path)


def make_submission_for_word2vec(ids, x_test, model, word2vec, n_features, idx2word, file_path):
    # 문제
    # 위의 코드를 사용해서 서브미션 파일을 만드세요.
    x_test = [s.lower().split() for s in x_test]  # text를 sequence 숫자로 변환
    features = [make_features_for_word2vec(
        tokens, word2vec, n_features, idx2word) for tokens in x_test]
    x_test = np.vstack(features)

    # x_test = lr.fit_transform(x_test)
    preds = model.predict(x_test)
    make_submission(ids, preds, file_path)

# 사용 모델 : word2vec, word2vec_nltk

def make_features_for_word2vec(tokens, word2vec, n_features, idx2word):
    binds , n_words  = np.zeros(n_features), 0
    # tokens:[the, in, happy, in]  인덱스 안에 들어있다면 딕셔너리 표기법, 숫자로 표기,
    for w in tokens:
        if w in idx2word:
            binds += word2vec.wv[w]
            n_words += 1

    return binds/(n_words if n_words > 0 else 1)
    # : RuntimeWarning: invalid value encountered in true_divide return binds/n_words


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

# gensim
def model_word2vec(x, y, ids, x_test):
    x = [s.lower().split() for s in x]

    n_features = 100
    word2vec = gensim.models.Word2Vec(
        x, workers=4, size=n_features,
        min_count=40, window=10, sample=0.001
    )
    print(word2vec)     # Word2Vec(vocab=9207, size=100, alpha=0.025)


    # print(word2vec.wv.index2word)           #
    # print(len(word2vec.wv.index2word))      # 534

    # list -> set : 검색 성능 향상
    # list 안에 features들이 들어가 있다.
    idx2word = set(word2vec.wv.index2word)

    features = [make_features_for_word2vec(
        tokens, word2vec, n_features, idx2word) for tokens in x]

    x = np.vstack(features) # 2차원으로 변환
    print(x.shape)      #

    # review 1 개 :
    # the, in, happy
    # the   : 1 0 0 0 0
    # in    : 0 0 1 0 0
    # happy : 0 0 0 1 0
    # in    : 0 0 1 0 0
    # ------------------
    #         1 0 2 1 0 / 4 원핫스타일

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data # submission 제출하는게 문제이므로 test는 없음
    # ------------------------------------------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight= 'balanced' )
    lr.fit(x_train, y_train)
    print('acc:', lr.score(x_valid, y_valid))
    # ------------------------------------------------------------------------- #

    # 문제
    # 위의 코드를 사용해서 서브미션 파일을 만드세요.
    # features = [make_features_for_word2vec(
    #         tokens, word2vec, n_features, idx2word) for tokens in x]
    #
    #     x = np.vstack(features) # 2차원으로 변환

    # 문제
    # 앞에서 사용한 전처리 코드를 사용해서 서브미션 파일을 만드세요.

    # x_test = lr.fit_transform(x_test)
    preds = lr.predict(x_test)
    make_submission(ids, preds, 'popcorn_model/word2vec.csv')

# nltk
def model_word2vec_nltk(x, y, ids, x_test):
    # x = [s.lower().split() for s in x]

    tokenizer = nltk.RegexpTokenizer(r'\w+')
    sent      = [tokenizer.tokenize(s.lower()) for s in x]
    # print(sent[1]) # ['the', 'classic', 'war', 'of', 'the', 'worlds',...

    st = nltk.stem.PorterStemmer()
    sent_stem = [[st.stem(w) for w in s] for s in sent]
    # stemming 후 다 흩어져버림
    # 문제

    stop_words = nltk.corpus.stopwords.words('english')
    sent_token = [[ w for w in s if w not in stop_words] for s in sent_stem]
    # print(sent_token[1]) # ['classic', 'war', 'world', 'timothi', 'hine', 'veri',

    x = sent_token

    # --------------------------------- #
    # 아래 코드는 model_word2vec 함수와 100% 동일

    n_features = 100
    word2vec = gensim.models.Word2Vec(
        x, workers=4, size=n_features,
        min_count=40, window=10, sample=0.001
    )
    print(word2vec)     # Word2Vec(vocab=9207, size=100, alpha=0.025)


    # print(word2vec.wv.index2word)           #
    # print(len(word2vec.wv.index2word))      # 534

    # list -> set : 검색 성능 향상
    # list 안에 features들이 들어가 있다.
    idx2word = set(word2vec.wv.index2word)

    features = [make_features_for_word2vec(
        tokens, word2vec, n_features, idx2word) for tokens in x]

    x = np.vstack(features) # 2차원으로 변환
    print(x.shape)      #

    # review 1 개 :
    # the, in, happy
    # the   : 1 0 0 0 0
    # in    : 0 0 1 0 0
    # happy : 0 0 0 1 0
    # in    : 0 0 1 0 0
    # ------------------
    #         1 0 2 1 0 / 4 원핫스타일

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data # submission 제출하는게 문제이므로 test는 없음
    # ------------------------------------------------------------------------- #

    lr = linear_model.LogisticRegression(class_weight= 'balanced' )
    lr.fit(x_train, y_train)
    print('acc:', lr.score(x_valid, y_valid))
    # ------------------------------------------------------------------------- #

    # make_submission_for_word2vec(
    #     ids, x_test, lr, word2vec, n_features, idx2word,
    #     'popcorn_model/word2vecnltk.csv')

    sent = [tokenizer.tokenize(s.lower()) for s in x]
    sent_stem = [[st.stem(w) for w in s] for s in sent]
    sent_token = [[w for w in s if w not in stop_words] for s in sent_stem]
    x_test = sent_stem
    #
    preds = lr.predict(x_test)
    make_submission(ids, preds, 'popcorn_model/word2vecnltk.csv')

def model_rnn(x, y, ids, x_test):
    vocab_size, seq_len = 2000, 200
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
    model.add(tf.keras.layers.Embedding(vocab_size, 100))  # 입력(2차원), 출력(3차원)
    # model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    # model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    cells = [tf.keras.layers.LSTMCell(64) for _ in range(2)]
    multi = tf.keras.layers.StackedRNNCells(cells)
    model.add(tf.keras.layers.RNN(multi))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, batch_size=128, epochs = 10, verbose=2,
              validation_data=(x_valid, y_valid))

    #----------------------------------------------------------------#
    make_submission_for_deep(ids, x_test, model, tokenizer, seq_len, 'popcorn_model/rnn.csv')


def model_cnn(x, y, ids, x_test):
    vocab_size, seq_len = 2000, 200
    x, tokenizer = tokenizing_and_padding(x, vocab_size, seq_len)

    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_valid, y_train, y_valid = data # submission 제출하는게 문제이므로 test는 없음
    # shuffle이 진행되면 순서가 바뀌어서 검증이
    # print(x.shape, y.shape)     # (1000, 200) (1000, 1)
    # ----------------------------------------------------------------------------- #
    # 문제
    # 학습하고 정확도를 구하는 나머지 코드를 만드세요.

    input = tf.keras.layers.Input(shape=[seq_len])

    embed = tf.keras.layers.Embedding(vocab_size, 100)(input)
    embed = tf.keras.layers.Dropout(0.5)(embed)

    conv1 = tf.keras.layers.Conv1D(128, 3, activation='relu')(embed)
    conv1 = tf.keras.layers.GlobalAvgPool1D()(conv1)

    conv2 = tf.keras.layers.Conv1D(128, 4, activation='relu')(embed)
    conv2 = tf.keras.layers.GlobalAvgPool1D()(conv2)

    conv3 = tf.keras.layers.Conv1D(128, 5, activation='relu')(embed)
    conv3 = tf.keras.layers.GlobalAvgPool1D()(conv3)

    concat = tf.keras.layers.concatenate([conv1, conv2, conv3])

    full1 = tf.keras.layers.Dense(256, activation='relu')(concat)
    full1 = tf.keras.layers.Dropout(0.5)(full1)

    full2 = tf.keras.layers.Dense(1, activation='sigmoid')(full1)

    model = tf.keras.Model(input, concat)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x_train, y_train, batch_size=128, epochs = 10, verbose=2,
              validation_data=(x_valid, y_valid))

    #----------------------------------------------------------------#
    make_submission_for_deep(ids, x_test, model, tokenizer, seq_len, 'popcorn_model/cnn.csv')

popcorn = pd.read_csv('popcorn/labeledTrainData.tsv',
                      delimiter='\t', index_col=0)
# print(popcorn)

x = popcorn.review.values
y = popcorn.sentiment.values.reshape(-1, 1)
# print(x.dtype, y.dtype)     # object int64

# token의 길이의 분포를 파악한후

n_samples = 1000
# x, y = x[:n_samples], y[:n_samples] # 에러나기전까지는 활성화 한다.
#
test_set = pd.read_csv('popcorn/testData.tsv',
                      delimiter='\t', index_col=0)

ids = test_set.index.values
x_test = test_set.review.values


# model_baseline(x, y, ids, x_test)
# model_tfidf(x, y, ids, x_test)
# model_word2vec(x, y, ids, x_test)
# model_word2vec_nltk(x, y, ids, x_test)
# model_rnn(x, y, ids, x_test)
model_cnn(x, y, ids, x_test)
# 1.baseline : loss: 0.1406 - acc: 0.9563 - val_loss: 0.9479 - val_acc: 0.6200
# 2.baseline_10 : loss: 0.0821 - acc: 0.9712 - val_loss: 0.4775 - val_acc: 0.8572

# baseline :
# tfidf    :
# word2vec : 0.8254
# rnn      :
# cnn      :


