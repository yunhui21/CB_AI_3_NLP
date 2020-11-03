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

