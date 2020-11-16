# Day_06_02_hamlet.py
import nltk
import collections
import matplotlib.pyplot as plt

# 문제
# 세익스피어의 햄릿에 등장하는 주인공들의 출현 빈도로 막대 그래프를 그려보세요
# gutenberg
# 햄릿, 거트루드, 오필리어, 클로디어스, 레어티스, 폴로니어스, 호레이쇼

# 1. 햄릿 읽기
print(nltk.corpus.gutenberg.fileids())      # 'shakespeare-hamlet.txt'

hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
hamlet = hamlet.lower()
tokens = nltk.regexp_tokenize(hamlet, r'\w+')
print(tokens[:10])

# 2. 등장 인물 영어 이름 찾기
actors = ['hamlet', 'gertrude', 'ophelia', 'claudius', 'laertes', 'polonius', 'horatio']
print([name in tokens for name in actors])

# 3. 빈도 계산
freq = collections.Counter(tokens)
print(freq['hamlet'])

values = [freq[name] for name in actors]
print(values)

# 4. 빈도 그래프 작성
plt.bar(actors, values)
plt.show()
