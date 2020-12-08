# Day_10_02_lambda.py


def twice(n):
    return n * 2


lamb = lambda n: n * 2


# 문제
# proxy 함수의 코드를 채우세요
def proxy(f, n):            # callback
    return f(n)


print(twice(3))
print(twice)

f = twice
print(f)
print(f(3))

print(proxy(twice, 7))
print(proxy(lamb, 7))
print(proxy(lambda n: n * 2, 7))

print(lamb(12))
print('-' * 50)

# 문제
# 1. 리스트를 오름차순 정렬하세요
# 2. 리스트를 내림차순 정렬하세요
a = [5, 9, 1, 3]

# a.sort()
# list.sort(a)
# b = sorted(a)
# print(a)
# print(b)

print(sorted(a))
print(sorted(a)[::-1])
print(sorted(a, reverse=True))


# 문제
# 1. colors를 오름차순 정렬하세요
# 2. colors를 내림차순 정렬하세요
def make_lower(s):
    return s.lower()


colors = ['Red', 'green', 'blue', 'YELLOW']
print(sorted(colors))
print(sorted(colors, reverse=True))

print(make_lower('ABC'))
print(make_lower('HeLLo'))

print(sorted(colors, key=make_lower))
print(sorted(colors, key=make_lower, reverse=True))

# 문제
# 람다를 사용해서 코드를 수정하세요
print(sorted(colors, key=lambda s: s.lower()))
print(sorted(colors, key=lambda s: s.lower(), reverse=True))

# 문제
# colors를 길이에 따라 정렬하세요 (람다 사용)
# 오름차순과 내림차순 적용하세요. 내림차순은 reverse 옵션을 사용하지 않습니다
print(sorted(colors, key=lambda s: len(s)))
print(sorted(colors, key=len))
print(sorted(colors, key=lambda s: len(s))[::-1])
print(sorted(colors, key=lambda s: -len(s)))
print('-' * 50)

# 문제
# 튜플로 구성된 리스트를 이름순 정렬, 나이순 정렬하세요
infos = [('kim', 25), ('han', 71), ('min', 37), ('nam', 13)]

print(sorted(infos))
print(sorted(infos, key=lambda t: t[0]))
print(sorted(infos, key=lambda t: t[1]))









