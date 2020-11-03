# Day_02_02_python.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.arange(6)
print(a)

print(a + 1)        # broadcast
print(a + a)        # vector operation , numpy에서만 양쪽의 개수가 같아야 한다.
print(np.sin(a))    # Universal function : sin함수를 배열에 포함된 개수만큼 적용

b = a.reshape(2, 3)
print(b)

print(b + 1)
print(b + b)
print(np.sin(b))

print(a[a > 1])     #데이터필터링을 위해서 사용할수있다.

# print(a + b)      # error
print(a.reshape(1, 6)) # 2차원
print(a.reshape(1, 6) + a)
print(a.reshape(6, 1))
print(a.reshape(6, 1) + a)  # (6,1) + (6,) => (6, 1) + (1, 6) 6행 6열이 나온다.
print()

print(a)
print(a[1:3], a[[1, 4]])    # []묶어주면 원하는 배열울 갖고 올 수 있다.
print()

print(b)
print(b[[1, 0, 1, 1]])
print(b[0][1], b[0, 1])     # fancy indexing, numpy
print(b[1][0], b[1, 0])
print()


# 문제
# 2차원 배열의 테두리를 1로 채우세요.
c = np.zeros([5, 5], dtype=np.int32) # np.int32 넘파이는 세분화 되어 있어서 np를 붙여야한다.
# c[0] , c[-1] = 1, 1
# c[:, 0] , c[:, -1]= 1
# c[[0, -1]] = 1
c[[0, -1], :] = 1
c[:, [0, -1]]= 1
print(c)

# 문제
# 2차원 배열의 양쪽 대각선을 2로 채우세요.
# c[0, 0] = 2
# c[1, 1] = 2
c[[0, 1, 2, 3,4], [0, 1, 2, 3,4]] = 2
c[range(5), list(reversed(range(5)))] = 2
print(c)

# c[0, [0, -1]] = 3
# c[1, [1, -2]] = 3
c[[0,1], [[0,1], [-1, -2]]] = 3
print()

d = [ 1, 4, 2, 7, 6, 9, 8]
d1 = np.random.choice(d, 5)
d2 = np.random.choice(d, 5)
print(d1)
print(d2)

plt.subplot(1, 2, 1)
plt.plot(range(len(d1)), d1, 'r')

# plt.figure() # 별도의 그래프가 그려진다.
plt.subplot(1, 2, 2)
plt.plot(range(len(d2)), d2, 'g')
# plt.show()

# data isris.csv, ai-times.tstory.com

# 문제
# iris 파일ㅇ르 읽어서 출력하세요.
iris = pd.read_csv('data/iris(150).csv', index_col=0)
iris.info()
print(iris)
# index, columns,:잘 사용하지 않는다,  values; 중요한 값이다.
print(iris.index)
print(iris.columns)
print(iris.values) # 넘퍼이다.

# 문제
# 데이터프레임에서 숫자와 문자열의 2개로 분ㄹ하세요.
x = iris.values[:, :-1]
print(x.shape)
y = iris.values[:, -1] # (150,)
y = iris.values[:, -1:] # (150,1) 가장 일반적
print(y.shape)


y = iris.Species # series
y = iris.Species.values # numpy로 변환
y = iris.Species.values.reshape(-1, 1)

iris2 = iris.drop(['Species'], axis=1)
print(iris2)

x = iris2.values
print(x.shape, y.shape)


