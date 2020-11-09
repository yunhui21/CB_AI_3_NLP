# Day_06_01_gensim.py
import gensim
import nltk
from pprint import pprint
import collections

# 일반 사람이 건강하거나 암에 걸렸을 비율을 계산해 봅시다. (건강한 사람과 암에 걸린 사람에 대한 확률)
# 암에 걸릴 확률: 0.1%라고 가정.
# 건강할 확률: 99.9%

# 암 판정을 받았을 때의 정확도: 95%
# 암이 아니라고 판정했을 때의 오진율: 2%

# 암 판정
# 실제 암에 걸렸을 확률: (암에 걸릴 확률 * 정확도) (암에 걸리지 않았을 확률 * 오진율)
# 0.001 * 0.95 = 0.00095 -> 0.045389 4.5%
# 0.999 * 0.02 = 0.01998 -> 0.954610 95%
# ----------------------
#                0.02093

# 문제
# 암이 아니라는 판정을 받았을때, 실제 암에 걸렸을 확률을 계산하세요.
# 암아 아니라고 판정 :
# 0.001 * 0.05 = 0.00005 -> 0.00005 0.1
# 0.999 * 0.98 = 0.97902 -> 0.99994 99%
# ----------------------
#                0.97907

def show_doc2vec():
    # 문제
    # computer.txt파일을 읽어서 문서별로 단어 토큰을 만드세요.
    f = open('data/computer.txt', 'r', encoding = 'utf-8')

    # 소문자 변환은 없지만, 마지막 빈줄을 삭제해야 함.
    # text = f.read()
    # text = text.lower()
    # text = text.split('\n')
    # print(len(text))
    # print(text[-1])
    # text.pop()

    # 마지막 빈줄은 없지만, lower를 줄 단위로 호출해야함.
    text = f.readlines()
    #print(len(text))

    docs = [d.lower().split() for d in text]
    print(*docs, sep = '\n')
    f.close()
    # ['graph', 'minors', 'a', 'survey']

    # 문제 2
    # docs로 부터 여러분의 불용어 목록을 만들어어 제거 하세요.
    stop_words = ['for', 'a', 'of', 'to', 'in', 'iv','and', 'the']
    docs = [[w for w in words if w not in stop_words] for words in docs]
    print(*docs, sep='\n')

    # [['human', 'machine', 'interface', 'lab', 'computer', 'applications'],
    # ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'],
    # ['user', 'interface', 'management', 'system'],
    # ['system', 'human', 'system', 'engineering', 'testing'],
    # ['relation', 'user', 'perceived', 'response', 'time', 'error', 'measurement'],
    # ['generation', 'random', 'binary', 'unordered', 'trees'],
    # ['intersection', 'graph', 'paths', 'trees'],
    # ['graph', 'minors', 'widths', 'trees', 'well', 'quasi', 'ordering'],
    # ['graph', 'minors', 'survey']]

    # 문제 3
    # 전체 문서에서 한 번만 출현한 토큰을 제거하세요.
    freq = collections.Counter([t for tokens in docs for t in tokens]) # list를 풀어주어야 한다.
    print(freq)
    print(freq['system'])
    docs = [[t for t in tokens if freq[t]>1] for tokens in docs]
    print(*docs, sep='\n')

    dot = gensim.corpora.Dictionary(docs) # 2차원이 들어가야 한다.
    print(dot)
    # Dictionary(12 unique tokens: ['computer', 'human', 'interface', 'response', 'survey']...)

    print(dot.token2id)
    # {'computer': 0, 'human': 1, 'interface': 2, 'response': 3, 'survey': 4,
    # 'system': 5, 'time': 6, 'user': 7, 'eps': 8, 'trees': 9, 'graph': 10,
    # 'minors': 11}

    print({v: k for k, v in dot.token2id.items()})
    print(dot.id2token) # 사용자가 직접 정으해서 사용하도록 한다.
    print()

    print(dot.doc2bow(['computer', 'trees', 'graph', 'trees'])) # 도큐먼트를 전달하는 코드
    # [(0, 1), (9, 2), (10, 1)] 앞쪽은 단어의 인덱스, 뒤쪽은 단어의 개수 12개 이상은 커질수없음.

    # 문제 4
    # docs에 포함된 문서를 bow(bag of words)로 변환하세요.
    print([dot.doc2bow(tokens) for tokens in docs], sep='\n')
    pprint([dot.doc2bow(tokens) for tokens in docs])
    '''
    [[(0, 1), (1, 1), (2, 1)],
     [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
     [(2, 1), (5, 1), (7, 1), (8, 1)],
     [(1, 1), (5, 2), (8, 1)],
     [(3, 1), (6, 1), (7, 1)],
     [(9, 1)],
     [(9, 1), (10, 1)],
     [(9, 1), (10, 1), (11, 1)],
     [(4, 1), (10, 1), (11, 1)]]
    '''

def show_doc2vec_1():
    text = ['나는 너를 사랑해', '나는 너를 미워해']
    token = [s.split() for s in text]
    print(token)

    # embedding = gensim.models.Word2Vec(token, min_count=1, size=5, sg=True)
    embedding = gensim.models.Word2Vec(token, min_count=1, size=5, sg=True)
    print(embedding) # Word2Vec(vocab=4, size=5, alpha=0.025) 유니크한 단어 4개, float타입 5개 사용,

    print(embedding.wv)
    print(embedding.wv.vectors)
    '''cbow
    [[-0.02725764 -0.04690436 -0.07834281  0.0814975  -0.05169397]
     [-0.07577898 -0.08840311 -0.0454245  -0.03986458  0.06979136]
     [ 0.02591396  0.06338769  0.08355726  0.075155   -0.09791408]
     [-0.06537742  0.02221168  0.01588749 -0.07072828  0.04705841]]
    '''
    '''sg (skip gram)결과가 더 좋아서 자주 사용한다.
    [[-0.0993235  -0.06465122  0.01683996 -0.05795937 -0.00530907]
    [-0.01578156  0.02876342  0.01885131  0.0227978   0.00126241]
    [ 0.08785617 -0.01244319 -0.00090108 -0.02748064  0.02600462]
    [-0.06032898  0.0105553   0.027974   -0.03355045 -0.08589888]]
    '''
    print(embedding.wv.vectors.shape)
    # (4, 5)
    print(embedding.wv['나는'])
    # [-0.04947044  0.00979599 -0.04026674  0.06795924  0.03817336]

# show_doc2vec()
show_doc2vec_1()