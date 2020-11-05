# Day_04_02_GenderClassification.py
import nltk
import random

# nltk.download('names')

# print(nltk.corpus.names.fileids())  # ['female.txt', 'male.txt']
# print(nltk.corpus.names.words())

# 문제
# 남자와 여자 이름으로 구성된 데이터셋을 반환하는 함수를 만드세요 (셔플 포함)
# 'kim', 'han'
# [('kim', 'male'), ('han', 'female'), ...]


def make_labeled_names():
    m = nltk.corpus.names.words('male.txt')
    f = nltk.corpus.names.words('female.txt')

    males = [(name, 'male') for name in m]
    females = [(name, 'female') for name in f]
    print(len(males), len(females))

    names = males + females
    random.shuffle(names)
    print(names[:5])

    return names


def make_feature_1(name):
    feature = {'last_letter': name[-1]}
    return feature


names = make_labeled_names()
data = [(make_feature_1(name), gender) for name, gender in names]

train_set, test_set = data[1000:], data[:1000]
# train_set, test_set = data[:7000], data[7000:]

clf = nltk.NaiveBayesClassifier.train(train_set)
print('acc :', nltk.classify.accuracy(clf, test_set))




