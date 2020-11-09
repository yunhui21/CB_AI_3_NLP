# Day_04_01_freq.py
import nltk
import collections
import operator
import matplotlib.pyplot as plt

# 문제
# webtext 코퍼스에 있는 wine 파일을 토큰으로 구성하세요
print(nltk.corpus.webtext.fileids())

wine = nltk.corpus.webtext.raw('wine.txt')
wine = wine.lower()
print(wine[:100])

tokens = nltk.regexp_tokenize(wine, r'\w+')
print(tokens[:10])

# 문제
# stopwords 코퍼스에 있는 english 파일을 토큰으로 구성하세요
# stopwords: 불용어
print(nltk.corpus.stopwords.fileids())

stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

# my_stops = ['나', '너']
# tokens = [w for w in tokens if w not in my_stops]

# 문제
# 토큰으로부터 불용어와 길이가 한 글자인 토큰을 제거하세요
tokens = [w for w in tokens if w not in stopwords]
tokens = [w for w in tokens if len(w) > 1]
print(tokens[:10])

# 문제
# 단어별 빈도를 딕셔너리에 저장하세요
# {'lovely': 163, 'delicate': 10, ...}


def make_freq_1(tokens):
    freq = {}
    for t in tokens:
        if t in freq:
            freq[t] += 1
        else:
            freq[t] = 1

    return freq


def make_freq_2(tokens):
    freq = {}
    for t in tokens:
        # if t in freq:
        #     freq[t] = freq.get(t) + 1
        # else:
        #     freq[t] = freq.get(t, 1)

        # if t in freq:
        #     freq[t] = freq.get(t, 0) + 1
        # else:
        #     freq[t] = freq.get(t, 0) + 1

        freq[t] = freq.get(t, 0) + 1

    return freq


def make_freq_3(tokens):
    # return [tokens.count(t) for t in tokens]      # bad
    # return [tokens.count(t) for t in set(tokens)]
    # return {tokens.count(t) for t in set(tokens)}
    return {t: tokens.count(t) for t in set(tokens)}


def make_freq_4(tokens):
    freq = collections.defaultdict(int)
    for t in tokens:
        freq[t] += 1

    return freq


# freq = make_freq_1(tokens)
# freq = make_freq_2(tokens)
# freq = make_freq_3(tokens)
# freq = make_freq_4(tokens)
# print(freq)

# print(freq['lovely'])
# print(freq.get('lovely'))
# print(freq['dont'])           # 예외 발생
# print(freq.get('dont'))       # None 반환
# print(freq.get('dont', 99))

# 문제
# freq 딕셔너리를 빈도에 따라 내림차순으로 정렬하세요
# [('good', 363), ('lovely', 163), ...]
# print(freq.items())
# print(sorted(freq.items()))
# print(sorted(freq.items(), key=lambda t: t[1]))
# print(sorted(freq.items(), key=lambda t: t[1], reverse=True))
# print(sorted(freq.items(), key=operator.itemgetter(1), reverse=True))

# freq = collections.Counter(tokens)
# print(freq)
# print(freq['good'])

freq = nltk.FreqDist(tokens)
print(freq)
print(freq['good'])
print(freq.N())
print(freq.most_common(10))
print(len(freq.most_common()))

# 문제
# 최빈값 5개를 막대 그래프로 그려보세요 (bar)
counts = [c for _, c in freq.most_common(5)]
words = [w for w, _ in freq.most_common(5)]
print(counts)

# plt.bar(range(5), range(5))

# plt.bar(range(5), counts)
# plt.xticks(range(5), words)

plt.bar(words, counts)
plt.show()
