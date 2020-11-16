# Day_07_02_NaverMovie.py
import csv
import re
import nltk
import collections
import matplotlib.pyplot as plt

# https://github.com/e9t/nsmc

# 문제 1
# x, y 데이터를 반환하는 함수를 만드세요

# 문제 2
# 김윤 박사 cnn sentence를 검색하면 깃헙이 나옵니다
# 해당 사이트에서 clean_str 함수를 찾아서 우리 코드에 적용하세요

# 문제 3
# train 데이터로 학습하고 test 데이터에 대해 정확도를 구하세요

# 문제 4
# 모든 문서의 토큰 갯수를 그래프로 그려보세요


def get_data(file_path):
    # 8544678	뭐야 아니잖아	0
    # f = open(file_path, 'r', encoding='utf-8')
    # # for row in f:
    # #     print(row.strip().split('\t'))
    #
    # for row in csv.reader(f, delimiter='\t'):
    #     print(row)
    #
    # f.close()

    f = open(file_path, 'r', encoding='utf-8')
    f.readline()

    x, y = [], []
    for _, doc, label in csv.reader(f, delimiter='\t'):
        # print(doc, label)
        x.append(clean_str(doc).split())
        y.append(label)

    f.close()

    # print(*x[:3], sep='\n')
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


def make_vocab(documents, vocab_size=2000):
    all_words = collections.Counter([w for doc in documents for w in doc])
    most_2000 = all_words.most_common(vocab_size)

    return [w for w, _ in most_2000]


def make_feature(doc_words, vocab):
    feature, uniques = {}, set(doc_words)

    for v in vocab:
        feature['has_{}'.format(v)] = (v in uniques)

    return feature


def make_feature_data(documents, labels, vocab):
    # return [(make_feature(documents[i], vocab), labels[i]) for i in range(len(labels))]
    return [(make_feature(doc, vocab), label) for doc, label in zip(documents, labels)]


def show_freq_dist(documents):
    print(len(documents))
    print(documents[0])
    print(max([len(doc) for doc in documents]))

    heights = [len(doc) for doc in documents]
    heights = sorted(heights)

    plt.plot(range(len(heights)), heights)
    plt.show()


# 문제 5
# 모든 문서의 토큰 길이를 25개로 맞춰주세요
# (25개를 넘어서는 만큼은 삭제하고, 부족한 부분은 공백 토큰으로 채웁니다)

def make_same_length(documents, max_len):
    new_docs = []
    for doc in documents:
        doc_len = len(doc)

        if doc_len < max_len:
            new_docs.append(doc + [''] * (max_len - doc_len))
        else:
            new_docs.append(doc[:max_len])

        # assert(len(new_docs[-1]) == max_len)

    return new_docs


x_train, y_train = get_data('data/ratings_train.txt')
x_test, y_test = get_data('data/ratings_test.txt')

# print(*x_train[:3], sep='\n')
# show_freq_dist(x_train)           # 토큰 40개 정도면 대부분의 리뷰를 포함

vocab = make_vocab(x_train, 2000)

x_train = make_same_length(x_train, max_len=25)
x_test = make_same_length(x_test, max_len=25)

train_set = make_feature_data(x_train, y_train, vocab)
test_set = make_feature_data(x_test, y_test, vocab)

clf = nltk.NaiveBayesClassifier.train(train_set)
print('acc :', nltk.classify.accuracy(clf, test_set))
