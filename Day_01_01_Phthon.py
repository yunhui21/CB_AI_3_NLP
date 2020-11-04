# Day_01_01_Phthon.py

'''
전반부
0. 리뷰
1. 자연어처리(머신러닝)
2. RNN입문(텐서플로)
3. 케라스 기초(딥러닝 리뷰)
4. RNN 활용

후반부
5. CNN 입문 + 활용
6. RNN/CNN 연동
7. 텐서플로 자격증
8. 텐서플로 허브, 케라스 튜너
9. RNN 고급 (seq2seq, 촛봇, attention)

목표
1. 데이터셋이 주어지면 알고리즘을 선택할 수 있다.
2. 해당 알고리즘의 코드를 수업 내용중에서 찾을 수 있다.
3. 새로운 데이터에 맞게 일부 코드를 수정할 수 있다.

4. 텐서플로 자격증 시험을 통과할 수 있다.
5. 시계열 데이터를 구성할 수 있는 코딩을 할 수 있다.

시작 : 5-10정도 review.
종료 : 20분정도 복습.
'''
# alt+1
# alt+4
# ctl+shift+f10
# ctl+c, ctl+v, ctl+x
# ctl+방향키 : drag
# tab, shift+tab
# ctl+/

print('hello, python')

# 리스트
# 문제 01
# 'hello'라는 문자열을 리스트로 만드세요.
a = 'hello'

a1 = []
for c in a:
    a1.append(c)
print(a1)

a2 = [c for c in a1]
print(a2)

a3 =list(a)
print(a3)

# 문제 02
# 숫자12345를 앞에서 만든 리스트에 연동하세요.
b = '12345'
print('hello' + '12345')
print(list('hello' + '12345'))
print(list('hello' + str(12345)))

# b1 = [c for c in b]
b1 = [int(i) for i in '12345']
c1 = a1 + b1
print(c1)

b = c1
# 문제 03
# 앞에서 만든 리스트를 거꾸로 뒤집으세요.

# 1.
print(b[::-1])
# print([i for i in range(len(b))])
# print([b[i] for i in range(len(b))])
# print([b[-i] for i in range(len(b))])
print([b[-i-1] for i in range(len(b))])

# print([b[len(b)-i -1] for i in range(len(b)))
# print([b[i] for i in reversed(range(len(b)))])

# list.reverse(b)
# b.reverse()
# print(b)

# 문제 04
# 'Red', 'green', 'BLUE' 가 들어있는 리스트를 자연스럽게 정렬하세요.
c =[ 'Red', 'green', 'BLUE']
# list.sort(c)
# print(c)
print(sorted(c))
print(sorted(c, reverse=True))
print(sorted(c, key=lambda s: s.lower())) # 값을 반환하는 한줄자리 함수 return은 안써도 된다. 무조건 반환한다고 사용하는것이라.]

# 문제
# 문자열이 포함된 리스트 C를 한 줄에 하나씩 출력하세요.
# 이때 문자열 앞에서 순서를 알려주는 숫자를 함게 출력합니다.
# 1
print(1, c[0])
print(2, c[1])
print(3, c[2])
# 2
print(c.count(c[0]), c[0])
print(c.count(c[1]), c[1])
print(c.count(c[2]), c[2])
# 3
i = 1
for s in c:
    print(i, s)
    i += 1
# 4
for s in enumerate(c):
    print(s)

for i, s in enumerate(c):
    print(i+1, s)

for i, s in enumerate(c, 1):
    print(i, s)

# 문제
# 리스트에 들어있는 문자열을 모두 거꾸로 뒤집으세요.
# 뒤집힌 문자열 리스트를 하나의 문자열로 변환하세요.


# 문제
# 아래 문장에 포함된 단얼르 거꾸로 뒤집어서 하나의 문장으로 만드세요.
# 'drum up the people'
# 'murd pu eht elpoep'
s = 'drum up the people'
print(s.split())
print([w[::-1] for w in s.split()])
print(' '.join([w[::-1] for w in s.split()]))