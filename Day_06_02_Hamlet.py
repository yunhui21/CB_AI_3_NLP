# Day_06_02_Hamlet.py

import nltk
import collections
import matplotlib.pyplot as plt
from matplotlib import colors
# 문제
# 세익스피어의 햄릿에 등장하는 주인공들의 출현 빈도로 막대 그래프를 그려보도록 합니다.
# gutenberg
# 햄릿, 거트루드, 오필리어, 클로디어스, 레이터스, 폴로니어스, 호레이쇼.
def hamlet():
    # print(nltk.corpus.gutenberg.fileids())
    hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
    # print(hamlet) # ['[', 'The', 'Tragedie', 'of', 'Hamlet', 'by', ...]
    hamlet = [w.lower() for w in hamlet]
    # print(hamlet) # ['[', 'the', 'tragedie', 'of', 'hamlet', 'by', ...]

    actor = ['hamlet','claudius','gertrude','polonius','ophelia','horatio','laertes']
    print([w in hamlet for w in actor])
    # [True, True, True, True, True, True, True]

    freq = nltk.FreqDist(hamlet)
    print(freq.most_common(5))
    # [(',', 2892), ('.', 1886), ('the', 993), ('and', 863), ("'", 729)]

    freq_actors = [freq[w] for w in actor]
    print(freq_actors)
    # [100, 1, 13, 20, 28, 40, 35]

    plt.bar(actor, freq_actors, color=colors.TABLEAU_COLORS)
    plt.show()

def hamlet_1():
    # 1. 햄릿 읽기
    #print(nltk.corpus.gutenberg.fileids()) #  'shakespeare-hamlet
    hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
    hamlet = hamlet.lower()
    tokens = nltk.regexp_tokenize(hamlet, r'\w+')
    print(tokens[:10]) # ['The', 'Tragedie', 'of', 'Hamlet', 'by', 'William', 'Shakespeare', '1599', 'Actus', 'Primus']

    # 2. 등장인물 영어이름 찾아서 맞는 이름인지 확인
    actor = ['hamlet','claudius','gertrude','polonius','ophelia','horatio','laertes']
    print([name in tokens for name in actor])

    # 3. 등장인물의 빈도수 확인 / 햄릿의 이름의 수를 확인
    freq = collections.Counter(tokens)
    print(freq['hamlet'])

    values = [freq[name] for name in actor]
    print(values)

    # 4. 빈도 그래프 형성
    plt.bar(actor, values)
    plt.show()

#hamlet()
hamlet_1()