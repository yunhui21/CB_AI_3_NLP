# practice.py
import numpy as np
import pandas as pd
import nltk



# 문제 01
# 'hello'라는 문자열을 리스트로 만드세요.
a = 'hello'
a = list(a)
# print(a)

# 문제 02
# 숫자12345를 앞에서 만든 리스트에 연동하세요.
b = '12345'
b = list(b)
c = a + b
# print(c)


# 문제 03
# 앞에서 만든 리스트를 거꾸로 뒤집으세요.
# print([w[::-1] for w in c])

# 문제 04
# 'Red', 'green', 'BLUE' 가 들어있는 리스트를 자연스럽게 정렬하세요.
d = ['Red', 'green', 'BLUE']
# print(' '.join(w for w in ))

# 문제 05
# 문자열이 포함된 리스트 C를 한 줄에 하나씩 출력하세요.
# 이때 문자열 앞에서 순서를 알려주는 숫자를 함게 출력합니다.
print()

# 문제 6
# 아래 문장에 포함된 단어를 거꾸로 뒤집어 하나의 문장으로 만드세요.
# 'drum up the people'
# 'murd pu eht elpoep'

# Day 2
# 1. 100보다 작은 양수가 12개 들어있는 리스트를 만드세요
#    (중복되어도 됩니다)

# 2. 100보다 작은 양수가 12개 들어있는 3행 4열의 리스트를 만드세요

# 3. 2차원 리스트를 1차원으로 변환하세요

# 4. 아래 문장에 포함된 가장 긴 단어의 길이를 알려주세요
#    f = 'the ventiation vane atop the building across the street spinning so fast says to me'

# 5. 앞에 사용했던 문장에서 아래 단어들을 제외한 단어장을 만드세요

# 6. 2차원 배열의 테두리를 1로 채우세요
#    c = np.zeros([5, 5], dtype=np.int32)

# 7. 2차원 배열의 양쪽 대각선을 2로 채우세요

# Day 3

# 문제
# 깔끔하세 알파벳 단어만으로 토큰을 구성하세요(2가지)
print(nltk.corpus.gutenberg)
print(nltk.corpus.gutenberg.fileids())
print(nltk.corpus.gutenberg)
moby = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')
moby = moby[:100]
print(moby)
token = nltk.tokenize.regexp_tokenize(moby, r['A-Za-z0-9가-힣']+))
print(token)
