# Day_04_01_freq.py
import nltk
import collections
import operator
import matplotlib.pyplot as plt

# 문제
# webtext 코퍼스에 있는 wine 파일을 토큰으로 구성하세요

# 문제
# 모든 토큰을 소문자로 만드세요
print(nltk.corpus.webtext.fileids())

wine = nltk.corpus.webtext.raw('wine.txt')
wine = wine.lower()
print(wine[:100])

tokens = nltk.regexp_tokenize(wine, r'\w+')
print(tokens[:10])

# 문제
# stopwords 코퍼스에 있는 english 파일을 토큰으로 구성하세요
# stopwords(불용어)
print(nltk.corpus.stopwords.fileids())

# stopwords = nltk.corpus.stopwords.raw('english')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

# 문제
# 와인 토큰으로부터 불용어와 한 글자로 된 토큰을 제거하세요
tokens = [t for t in tokens if t not in stopwords]
tokens = [t for t in tokens if len(t) > 1]
print(tokens[:100])

# 문제
# 단어별 빈도를 딕셔너리에 저장하세요
# {'lovely': 163, 'delicate': 10, ...}
def freq_1(tokens):
    freq = {}
    for t in tokens:
        if t in freq:
            freq[t] += 1
        else:
            freq[t] = 1

    return freq


# 문제
# 딕셔너리의 get 함수를 이용하도록 수정하세요
def freq_2(tokens):
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    return freq


def freq_3(tokens):
    # return [tokens.count(t) for t in tokens]      # bad
    # return [tokens.count(t) for t in set(tokens)]
    # return {tokens.count(t) for t in set(tokens)}
    return {t: tokens.count(t) for t in set(tokens)}


def freq_4(tokens):
    freq = collections.defaultdict(int)
    for t in tokens:
        freq[t] += 1

    return freq


# print(freq_1(tokens))
# print(freq_2(tokens))
# print(freq_3(tokens))
# print(freq_4(tokens))
# {'lovely': 163, 'delicate': 10, 'fragrant': 45, ...}

# freq = freq_3(tokens)
# print(freq['fragrant'], freq.get('fragrant'))
# print(freq.get('frog'), freq['frog'])     # None error
# print(freq.get('frog', 1))

# 문제
# freq 딕셔너리를 빈도에 따라 정렬된 리스트로 변환하세요 (내림차순)
# print(list(freq.items()))
# print(sorted(freq.items()))
# print(sorted(freq.items(), key=lambda t: t[1]))
# print(sorted(freq.items(), key=lambda t: t[1], reverse=True))
# print(sorted(freq.items(), key=operator.itemgetter(1), reverse=True))

freq = nltk.FreqDist(tokens)
print(freq)
print(freq.N())
print(freq.most_common(10))
print(freq['good'])
print(len(freq.most_common()))

# freq = collections.Counter(tokens)
# print(freq)

# 문제
# 최빈값 5개를 막대 그래프로 그리세요 (bar)
# counts = [c for _, c in freq.most_common(5)]
# words = [w for w, _ in freq.most_common(5)]

# [('good', 363), ('quite', 303), ('fruit', 300), ('wine', 234), ('bit', 217)]
# ('good', 363), ('quite', 303), ('fruit', 300), ('wine', 234), ('bit', 217)
words, counts = list(zip(*freq.most_common(5)))

# plt.bar(range(5), range(5))
# plt.bar(range(5), counts)
# plt.xticks(range(5), words)

plt.bar(words, counts)
plt.show()


