# Day_06_01_gensim.py
import gensim
import nltk
import collections
from pprint import pprint


# 일반 사람이 건강하거나 암에 걸렸을 비율을 계산해 봅시다
# 암에 걸릴 확률: 0.1%
# 건강할 확률: 99.9%

# 암 판정을 받았을 때의 정확도: 95%
# 암이 아니라고 판정했을 때의 오진율: 2%

# 암 판정
# 실제 암에 걸렸을 확률: (암에 걸릴 확률 * 정확도) (암에 걸리지 않았을 확률 * 오진율)
# 0.001 * 0.95 = 0.00095 -> 0.04538939321548
# 0.999 * 0.02 = 0.01998 -> 0.95461060678452
# ----------------------
#                0.02093

# 문제
# 암이 아니라는 판정을 받았을 때, 실제 암에 걸렸을 확률을 계산하세요
# 0.001 * 0.05 = 0.00005 -> 0.00005106887148
# 0.999 * 0.98 = 0.97902 -> 0.99994893112852
# ----------------------
#                0.97907


def show_doc2vec():
    # 문제 1
    # computer.txt 파일을 읽어서 문서별로 단어 토큰을 만드세요
    # [['Human', 'machine', 'interface', ...], ['A', 'survey', 'of'], ...]
    f = open('data/computer.txt', 'r', encoding='utf-8')

    # 소문자 변환은 쉽지만, 마지막 빈줄을 삭제해야 함
    # text = f.read()
    # text = text.lower()
    # text = text.split('\n')
    # print(len(text))
    # print(text[-1])
    # text.pop()
    # print(len(text))

    # 마지막 빈줄은 없지만, lower를 줄 단위로 호출해야 함
    text = f.readlines()
    print(len(text))

    docs = [d.lower().split() for d in text]
    # print(*docs, sep='\n')

    f.close()

    # 문제 2
    # 여러분의 불용어 목록을 만들어서 docs로부터 제거하세요
    my_stopwords = ['a', 'the', 'of', 'and', 'in', 'for', 'to']
    docs = [[t for t in tokens if t not in my_stopwords] for tokens in docs]
    print(*docs, sep='\n')

    # 문제 3
    # 전체 문서에서 한 번만 출현한 토큰을 제거하세요
    freq = collections.Counter([t for tokens in docs for t in tokens])
    print(freq)
    print(freq['system'])

    docs = [[t for t in tokens if freq[t] > 1] for tokens in docs]
    print(*docs, sep='\n')

    dct = gensim.corpora.Dictionary(docs)
    print(dct)
    # Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)

    print(dct.token2id)
    # {'computer': 0, 'human': 1, 'interface': 2, 'response': 3, 'survey': 4,
    # 'system': 5, 'time': 6, 'user': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}

    print({v: k for k, v in dct.token2id.items()})
    print(dct.id2token)
    print()

    print(dct.doc2bow(['computer', 'trees', 'graph', 'trees']))     # 도큐먼트 전달
    # [(0, 1), (9, 2), (10, 1)]

    # 문제 4
    # docs에 포함된 문서를 bow(bag of words)로 변환하세요
    print(*[dct.doc2bow(tokens) for tokens in docs], sep='\n')
    pprint([dct.doc2bow(tokens) for tokens in docs])


def show_word2vec_1():
    text = ['나는 너를 사랑해', '나는 너를 미워해']
    token = [s.split() for s in text]
    print(token)

    # embedding = gensim.models.Word2Vec(token, min_count=1, size=5)
    embedding = gensim.models.Word2Vec(token, min_count=1, size=5, sg=True)
    print(embedding)

    print(embedding.wv)
    print(embedding.wv.vectors)
    print(embedding.wv.vectors.shape)

    print(embedding.wv['나는'])
    print(embedding['나는'])


def show_word2vec_2():
    r1 = nltk.corpus.movie_reviews.raw('neg/cv000_29416.txt')       # str
    r2 = nltk.corpus.movie_reviews.words('neg/cv000_29416.txt')     # 1차원 리스트
    r3 = nltk.corpus.movie_reviews.sents('neg/cv000_29416.txt')     # 2차원 리스트

    print(type(r1), r1[:5])
    print(type(r2), r2[:5])
    print(type(r3), r3[:5])

    sents = nltk.corpus.movie_reviews.sents()
    model = gensim.models.Word2Vec(sents)

    # 코사인 유사도: -1 ~ 1 (실제로는 0 ~ 1)
    print(model.wv.similarity('villain', 'hero'))
    print(model.wv.similarity('man', 'woman'))
    print(model.wv.similarity('sky', 'earth'))
    print(model.wv.similarity('water', 'iron'))

    print(model.wv.most_similar('apple'))
    # [('raging', 0.9470874071121216), ('empire', 0.9344267845153809),
    # ('waters', 0.9318031668663025), ('alliance', 0.9298103451728821),
    # ('mafia', 0.9290547966957092), ('sand', 0.9285709261894226),
    # ('cloud', 0.9284428358078003), ('iron', 0.9281488656997681),
    # ('bull', 0.927939772605896), ('1984', 0.9275203943252563)]

    # print('papa' in model)
    print('papa' in model.wv)

    print(model.wv['apple'])
    print(model.wv['apple'].shape)


# show_doc2vec()
# show_word2vec_1()
show_word2vec_2()
