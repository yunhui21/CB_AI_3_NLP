# Day_03_02_Nltk.py
# nltk,
import nltk # natural language Tool kit


# 데이터 -> 코퍼스(도메인, 말뭉치)(corpus) -> 토큰화(tokenizer)  -> 어간추출(steming) -> 품사 태깅 (pos)

def load_datasets():
    nltk.download('gutenberg')
    nltk.download('webtext')
    nltk.download('reuters')
    nltk.download('punkt')

    nltk.download()

def corpus():
    print(nltk.corpus.gutenberg)

    print(nltk.corpus.gutenberg.fileids())
    '''
    ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 
    'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 
    'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 
    'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 
    'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 
    'shakespeare-macbeth.txt', 'whitman-leaves.txt']
    '''
    print(nltk.corpus.gutenberg)
    moby = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
    print(moby[:1000])
    print(type(moby))

    words = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
    # ['[', 'Moby', 'Dick', 'by', 'Herman', 'Melville', '1851', ']', ...']']
    print(words[:100])

    print(len(moby), len(words)) # 1242990 260819

def tokenize():
    moby = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
    moby = moby[:100]
    print(nltk.tokenize.simple.SpaceTokenizer()) # \r\n\r\n\r\n 처리 못함
    print(nltk.tokenize.simple.SpaceTokenizer().tokenize(moby)) # 문장 전달.
    # ['[Moby', 'Dick', 'by', 'Herman', 'Melville', '1851]\r\n\r\n\r\nETYMOLOGY
    print(nltk.tokenize.sent_tokenize(moby))

    # moby_sent =[]
    for sent in nltk.tokenize.sent_tokenize(moby):
        print(sent)
        print('---')
        # moby_sent.append(sent)
    # print(moby_sent)


    # 문제
    # 깔끔하세 알파벳 단어만으로 토큰을 구성하세요(2가지)
    # token1 = nltk.tokenize.simple.SpaceTokenizer().tokenize(sent)
    # print(token1)
    # token2 = nltk.tokenize.word_tokenize().tokenize(sent)

    # token = nltk.tokenize.word_tokenize(moby)
    # token = nltk.tokenize.wordpunct_tokenize(moby)
    # token = nltk.tokenize.WordPunctTokenizer().tokenize(moby)

    print([t for t in nltk.tokenize.word_tokenize(moby) if t not in ['[',']']])
    print(nltk.tokenize.regexp_tokenize(moby, r'\w+'))

    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Z]'))
    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Za-z]')) # 한글자씩만 찾기
    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Za-z]+')) # 영문단어 찾기
    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Za-z0-9]+')) # 숫자 찾기
    print(nltk.tokenize.regexp_tokenize(moby, r'[A-Za-z0-9가-힣]+')) # 한글 찾기


# load_datasets()
# corpus()
tokenize()
