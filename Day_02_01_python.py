# Day_02_01_python.py
import random

print(random.randrange(10, 20, 3))      # 10 13 16 19

# 문제
# 100보다 작은 양수가 12개 들어있는 3행 4열의 리스트를 만드세요
# (컴프리헨션 사용하면 칭찬합니다)

# a = []
# for i in range(3):
#     # b = []
#     # for j in range(4):
#     #     print(i, j)
#     #     b.append(random.randrange(100))
#     # a.append(b)
#     a.append([random.randrange(100) for _ in range(4)])

a = [[random.randrange(100) for _ in range(4)] for _ in range(3)]
print(a)

# 1차원이기 때문에 2차원으로 변환 필요
# print(random.sample(range(1, 100), 12))
# print([random.randrange(100) for _ in range(12)])

# 문제
# 2차원 리스트를 1차원으로 변환하세요
b = [j for i in a for j in i]
# b = [j for _ in a for j in _]     # 사용할 수 있지만 나쁜 코드
print(b)

for i in a:
    # print(i)
    for j in i:
        print(j, end=' ')
print()

# 문제
# 아래 문장에 포함된 가장 긴 단어의 길이를 구하세요
c = 'the ventiation vane atop the building across the street spinning so fast says to me'

max_len = 0
for w in c.split():
    print(len(w), w)

    if max_len < len(w):
        max_len = len(w)

print(max_len)

print([len(w) for w in c.split()])
print(max([len(w) for w in c.split()]))
print()

# 문제
# 위의 문장으로부터 아래 단어를 제외한 단어들로 이루어진 단어장을 만드세요
filtered = ['the', 'across', 'to', 'a', 'in']

for w in c.split():
    if len(w) == 4:
        print(w)

print([w for w in c.split() if len(w) == 4])
print([w for w in c.split() if w in filtered])
print([w for w in c.split() if w not in filtered])

d = [w for w in c.split() if w not in filtered]
print(d.index('vane'), d.index('building'))
