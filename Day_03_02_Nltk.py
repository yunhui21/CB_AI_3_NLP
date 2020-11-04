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

def stemming():
    words = ['lives', 'dies', 'flies', 'died']

    st = nltk.stem.PorterStemmer() #
    print(st.stem('lives'))

    # 문제
    # words 에 포함된 모든 단어의 어근(어간)을 출력하세요.
    print([w for w in words])
    print([st.stem(w) for w in words]) # ['live', 'die', 'fli', 'die']

    st = nltk.stem.LancasterStemmer() # porterstemmer()
    print([st.stem(w) for w in words]) # ['liv', 'die', 'fli', 'died']

def grams():
    text = 'you deliver that which is not true'
    tokens = nltk.tokenize.word_tokenize(text)
    print(tokens) # ['you', 'deliver', 'that', 'which', 'is', 'not', 'true']

    # book desk : 관련성을 다질 수 있는가? 동시발생을 따져 본다.

    # 문제
    # 토큰에서 3단어씩 순서대로 묶어 보세요.
    # ['you', 'deliver', 'that', 'which', 'is', 'not', 'true']
    # ('you', 'deliver', 'that')('deliver', 'that', 'which')...5개
    # g = [i for i in range(len(tokens)-2)]
    # g = [(tokens[i],tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
    g = [(tokens[i:i+3]) for i in range(len(tokens)-2)]
    print(g)
    print(*g)
    print(*g, sep = '\n')
    print('-'*15)

    print(tuple(nltk.bigrams(tokens)))   # 2개
    print(tuple(nltk.trigrams(tokens)))  # 3개
    print(tuple(nltk.ngrams(tokens, 5)), sep='\n') # 5개

# load_datasets()
# corpus()
# tokenize()
# stemming()
grams()

# 빈도파악, 피쳐엔지니어링