# Day_05_01_doc2vec.py
import nltk
import random
import collections
import time


def make_vocab(vocab_size=2000):
    # nltk.download('movie_reviews')

    words = nltk.corpus.movie_reviews.words()
    # print(len(words))           # 1583820
    #
    # print(nltk.corpus.movie_reviews.categories())   # ['neg', 'pos']
    # print(nltk.corpus.movie_reviews.fileids('neg'))
    # print(nltk.corpus.movie_reviews.fileids('pos'))
    #
    # print(len(nltk.corpus.movie_reviews.fileids('neg')))    # 1000
    # print(len(nltk.corpus.movie_reviews.fileids('pos')))    # 1000

    all_words = collections.Counter([w.lower() for w in words])
    most_2000 = all_words.most_common(vocab_size)

    return [w for w, _ in most_2000]


def make_feature(filename, vocab):
    doc_words = nltk.corpus.movie_reviews.words(filename)
    feature, uniques = {}, set(doc_words)

    for v in vocab:
        feature['has_{}'.format(v)] = (v in uniques)

    return feature


vocab = make_vocab(vocab_size=2000)

# 파일 목록
neg = nltk.corpus.movie_reviews.fileids('neg')
pos = nltk.corpus.movie_reviews.fileids('pos')

# 문제
# 80% 데이터로 학습하고 20% 데이터에 대해 정확도를 구하세요
neg_data = [(make_feature(filename, vocab), 'neg') for filename in neg]
pos_data = [(make_feature(filename, vocab), 'pos') for filename in pos]

# 긍정/부정 불균형
# data = neg_data + pos_data
# random.shuffle(data)

random.shuffle(neg_data)
random.shuffle(pos_data)

train_set = neg_data[:800] + pos_data[:800]
test_set = neg_data[800:] + pos_data[800:]

clf = nltk.NaiveBayesClassifier.train(train_set)
print('acc :', nltk.classify.accuracy(clf, test_set))
