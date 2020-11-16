# Day_03_02_nltk.py
import nltk         # natural language toolkit

# 데이터 -> 코퍼스 -> 토큰화 -> 어간 -> 품사
# 사람은, 사람이, 사람에게, 사람의, 사람이라면, 사람일까, ...


def load_datasets():
    nltk.download('gutenberg')
    nltk.download('webtext')
    nltk.download('reuters')

    # nltk.download()


def corpus():
    print(nltk.corpus.gutenberg)

    print(nltk.corpus.gutenberg.fileids())
    print(nltk.corpus.gutenberg)

    moby = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
    print(moby[:1000])
    print(type(moby))

    words = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
    print(words)


def tokenize():
    moby = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
    moby = moby[:1000]

    print(nltk.tokenize.sent_tokenize(moby))

    for sent in nltk.tokenize.sent_tokenize(moby):
        print(sent)
        print('------------')

    print(nltk.tokenize.word_tokenize(moby))        # fail
    print(nltk.tokenize.wordpunct_tokenize(moby))   # fail

    print(nltk.tokenize.regexp_tokenize(moby, r'\w+'))

    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Z]'))
    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Za-z0-9]+'))
    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Za-z0-9가-힣]+'))


def stemming():
    words = ['lives', 'dies', 'flies', 'died']

    st = nltk.stem.PorterStemmer()
    print(st.stem('lives'))

    # 문제
    # words 전체에 대해 어간을 추출하세요
    print([st.stem(w) for w in words])

    st = nltk.stem.LancasterStemmer()
    print([st.stem(w) for w in words])


def grams():
    text = 'all the known nations of the world'
    tokens = nltk.word_tokenize(text)
    # tokens = nltk.tokenize.word_tokenize(text)
    print(tokens)
    print()

    # 문제
    # 토큰에서 3단어씩 순서대로 묶어주세요
    # ('all', 'the', 'known') ('the', 'known', 'nations') ...
    # print([i for i in range(len(tokens)-2)])
    # print([tokens[i] for i in range(len(tokens)-2)])
    # print([(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)])
    # print([tokens[i:i+3] for i in range(len(tokens)-2)])

    g = [tokens[i:i+3] for i in range(len(tokens)-2)]
    print(g)
    print(*g)
    print(*g, sep='\n')
    print()

    print(tuple(nltk.bigrams(tokens)))
    print(list(nltk.trigrams(tokens)))
    print(*list(nltk.ngrams(tokens, 5)), sep='\n')
    print()

    print([(0, 1), (2, 3), (4, 5)])
    print(*[(0, 1), (2, 3), (4, 5)])
    print((0, 1), (2, 3), (4, 5))
    print(*[(0, 1), (2, 3), (4, 5)], sep='**')
    print((0, 1), (2, 3), (4, 5), sep='**')


# load_datasets()
# corpus()
# tokenize()
# stemming()
# grams()
