# Day_02_01_python.py

import random


random.seed(23)# random값 고정
# 문제
# 100보다 작은 양수가 12개 들어있는 리스트를 만드세요.
# 1.
# a = [i for i in range(1, 100, 10)]
a = [0 for _ in range(12)]
a = [[0] for _ in range(12)]
a = [random.randrange(100) for _ in range(12)]
print(a)

# 문제
# 100보다 작은 양수가 12개 들어있는 3행 4열의 리스트를 만드세요. # 2차원
# 1.
b = [[random.randrange(100) for _ in range(4)] for _ in range(3)]
# 2.
b = []
for i in range(3):
    c = []
    for j in range(4):
        print(i, j)
        c.append(random.randrange(100))
    b.append(c)
print(b)

# 문제
# 2차원 리스트를 1차원으로 변환하세요.
# enumerate, iterable,
e = []
for i in b:
    for j in i:
        print(j)
        e.append(j)

f = [j for i in b for j in i]
print(f)

# 문제
# 아래 문장에서 포함된 가장 긴 단어의 길이를 알려주세요.
f = 'Who but is pleased to watch the moon on high and nothing loth her Majesty'

max_len = 0
for s in f.split():
    print(len(s), s)

    if max_len < len(s):
        max_len = len(s)
print(max_len)
print([len(s) for s in f.split()])
print(max([len(s) for s in f.split()]))

# 문제
# 앞에 사용했던 문장에서 아래 단어들을 제외한 단어로 단어장을 만드세요.
filter = ['is','to', 'on', 'loth']

print( [s for s in f.split() if len(s) == 4])
print( [s for s in f.split() if s not in filter])

g = [s for s in f.split() if s not in filter]
# print(g) # 'but' 'watch'
print(g.index('but'), g.index('watch')) #exception이 발생하는 경우에 대해서 처리를 하지 않으면 down된다.

