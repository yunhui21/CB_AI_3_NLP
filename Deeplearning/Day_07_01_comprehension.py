# Day_07_01_comprehension.py
import random

# 리스트내포

for i in range(10):
    i

a = []
for i in range(10):
    a.append(i)

print(a)

[i for i in range(10)]      # 리스트
(i for i in range(10))      # 튜플
{i for i in range(10)}      #

print([i for i in range(10)])
print(sum([i for i in range(10)]))

for i in range(5):
    print(random.randrange(100), end=' ')
print()

# 문제
# 100보다 작은 난수가 10개 들어있는 리스트를 만드세요
b = [random.randrange(100) for _ in range(10)]  # place holder
print(b)

# 문제
# 리스트에서 홀수만 뽑아서 별도의 리스트를 만드세요 (컴프리헨션)
a = []
for i in b:
    if i % 2:
        a.append(i)
        print(i, end=' ')
print()
print(a)
print([i for i in b if i % 2])

a1 = [random.randrange(100) for _ in range(10)]
a2 = [random.randrange(100) for _ in range(10)]
a3 = [random.randrange(100) for _ in range(10)]

c = [a1, a2, a3]
# c = [a1, a2, a3, a1, a2, a3]

# 문제
# 2차원 리스트의 전체 합계를 구하세요
print(sum(a1) + sum(a2) + sum(a3))
print(sum(a1), sum(a2), sum(a3))
print([sum(a1), sum(a2), sum(a3)])
print(sum([sum(a1), sum(a2), sum(a3)]))
print([sum(i) for i in c])
print(sum([sum(i) for i in c]))
print(sum((sum(i) for i in c)))
print(sum(sum(i) for i in c))

t1 = [sum(i) for i in c]
t2 = (sum(i) for i in c)
# t2[0] = 99
print('-' * 50)

# 문제
# 2차원 리스트를 1차원 리스트로 변환하세요
# (반복문으로 먼저 구성한 후에 진행해 봅니다)
d = []
for i in c:
    # print(i)
    for j in i:
        d.append(j)
        print(j, end=' ')
print()
print(d)
print([j for i in c for j in i])
print('-' * 50)

# 문제 (구글 입사)
# 1 ~ 10000 사이의 정수에 포함된 8의 갯수를 구하세요
# 808 -> 2
# 힌트 : 문자열 클래스에 포함된 어떤 함수를 사용합니다
def count_8(n):
    a1 = n // 1 % 10 == 8       # 808 //    1 = 808 % 10 = 8
    a2 = n // 10 % 10 == 8      # 808 //   10 =  80 % 10 = 0
    a3 = n // 100 % 10 == 8     # 808 //  100 =   8 % 10 = 8
    a4 = n // 1000 % 10 == 8    # 808 // 1000 =   0 % 10 = 0

    return a1 + a2 + a3 + a4

print(count_8(1))
print(count_8(81))
print(count_8(808))
print(count_8(8888))
print(count_8(1357))

print([i for i in range(10)])
print([count_8(i) for i in range(10)])
print(sum([count_8(i) for i in range(10000)]))

# int, float, bool, str
print([str(i) for i in range(10)])
print([str(i).count('8') for i in range(10)])
print(sum([str(i).count('8') for i in range(10000)]))

# 차원 dimension
# 0차원 7
# 1차원 [1, 4, 7]
#          ^
# 2차원 [[1, 4, 7],
#       [2, 3, 9]]    <
#           ^
# 3차원 [[[1, 4, 7], [2, 3, 9]], [[1, 4, 7], [2, 3, 9]]]







