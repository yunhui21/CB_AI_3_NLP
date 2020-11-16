# Day_01_01_python.py

# applekoong@naver.com 김정훈

# 자연어 처리 (32)
# RNN (28)
# 제조 데이터 & 챗봇 구현 (40)
# 프로젝트 (56)


# 0. 파이썬 리뷰
# 1. 자연어 처리 (머신러닝)
# 2. RNN 입문 (텐서플로)
# 3. 케라스 기초 (딥러닝 리뷰)
# 4. RNN 활용
# 5. CNN 입문 + 활용
# 6. RNN/CNN 연동
# 7. 텐서플로 자격증
# 8. 텐서플로 허브, 케라스 튜너
# 9. RNN 고급 (seq2seq, 챗봇, attention)


# 목표
# 1. 데이터셋이 주어지면 알고리즘을 선택할 수 있다
# 2. 해당 알고리즘의 코드를 수업 내용 중에서 찾을 수 있다
# 3. 새로운 데이터에 맞게 일부 코드를 수정할 수 있다

# 4. 텐서플로 자격증을 취득한다
# 5. 시계열 데이터를 구성할 수 있는 코딩을 할 수 있다

# 환경 구축
# 1. 파이썬 설치 (3.7, 64비트)
# 2. 파이참 설치 (커뮤니티)

# ctrl + shift + f10 (윈도우)
# ctrl + shift + R (맥북)

# alt + 1 (윈도우)  cmd + 1 (맥북)
# alt + 4 (윈도우)  cmd + 4 (맥북)

# ctrl(cmd)키를 누른 상태에서 c, v, v 연속 타격
# ctrl(cmd) + /

# ---------------------------------------------------- #

# 리스트

# 문제
# 'hello'라는 문자열을 리스트로 만드세요 (2가지)

a = 'hello'

a1 = []
for i in a:
    a1.append(i)
print(a1)

a2 = [i for i in a]         # 컴프리헨션
print(a2)

a3 = list(a)
print(a3)
print()

# 문제
# 숫자 12345에 포함된 각각의 숫자를 앞에서 만든 리스트에 연동하세요
# (리스트에 포함된 요소 갯수가 10개라는 뜻입니다)
print('hello' + '12345')
print(list('hello' + '12345'))
print(list('hello' + str(12345)))

# a3.extend([1, 3, 5])
a3 += [1, 3, 5]
print(a3)

print([i for i in str(12345)])
print([int(i) for i in str(12345)])
print()

# h, 1, e, 2, l, 3, l, 4, o, 5 와 같이 출력하려면 어떻게 해야할까요?
# a = 'hello'
# a2 = list(a)
# c = '12345'
# c2 = list(c)
#
# d = list(zip(a2, c2))
# print([j for i in d for j in i])

b = a1 + list(str(12345))
print(b)

# 문제
# 앞에서 만든 리스트를 거꾸로 뒤집으세요 (2가지)
print(b[::-1])

# list.reverse(b)
# print(b)

print([i for i in range(len(b))])
print([i for i in reversed(range(len(b)))])
print([b[i] for i in reversed(range(len(b)))])
print()

print([b[len(b) - i - 1] for i in range(len(b))])
print([b[-i] for i in range(len(b))])       # wrong
print([b[-i-1] for i in range(len(b))])

print(reversed(b))
print(list(reversed(b)))
print()

# 문제
# ['Red', 'green', 'Blue'] 리스트를 자연스럽게 정렬하세요
c = ['Red', 'green', 'Blue']

# c.sort()
# list.sort(c)
# print(c)


def my_lower(s):
    return str.lower(s)


print(sorted(c))
print(sorted(c, reverse=True))
print(sorted(c, key=lambda w: w.lower()))
print(sorted(c, key=lambda w: str.lower(w)))
print(sorted(c, key=my_lower))
print(sorted(c, key=str.lower))
print()

# 문제
# ['Red', 'green', 'Blue'] 리스트를 아래처럼 출력하세요
# 1 Red
# 2 green
# 3 Blue

i = 1
for s in c:
    print(i, s)
    i += 1
print()

for s in enumerate(c):
    print(s[0]+1, s[1])
print()

for i, s in enumerate(c, 1):
    print(i, s)
print()

for i in enumerate(enumerate(c, 1)):
    print(i)
print()

for i0, (i1, s) in enumerate(enumerate(c, 1)):
    print(i0, i1, s)
print()

# 문제
# 아래 문장에 포함된 단어를 거꾸로 뒤집어서 하나의 문장으로 만드세요
# 원본: 'drum up the people'
# 결과: 'murd pu eht elpoep'
s = 'drum up the people'

print(s.split())
print([w for w in s.split()])
print([w[::-1] for w in s.split()])
print(' '.join([w[::-1] for w in s.split()]))

t = [w[::-1] for w in s.split()]
print(' '.join(t))
