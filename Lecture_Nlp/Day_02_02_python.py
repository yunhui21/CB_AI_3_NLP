# Day_02_02_python.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.arange(6)
print(a)

print(a + 1)                # broadcast
print(a + a)                # vector operation
print(np.sin(a))            # universal function

b = a.reshape(2, 3)
print(b)

print(b + 1)                # broadcast
print(b + b)                # vector operation
print(np.sin(b))            # universal function

print(a > 1)
print(a[a > 1])
print()

# a + b         # (6,) + (2, 3)

print(a.reshape(1, 6))
print(a.reshape(1, 6) + a)      # (1, 6) + (6,)

print(a.reshape(6, 1))
print(a.reshape(6, 1) + a)      # (6, 1) + (1, 6)
print()

print(a)
print(a[0], a[3], a[1:3])
print(a[[1, 4]])

print(b)
print(b[[1, 0, 1, 1]])
print()

print(b[0][2], b[0, 2])         # fancy indexing
print(b[1][0], b[1, 0])
print()

# 문제
# 2차원 배열의 테두리를 1로 채우세요
c = np.zeros([5, 5], dtype=np.int32)
# c[0], c[-1] = 1, 1
# c[0, :], c[-1, :] = 1, 1
# c[:, 0], c[:, -1] = 1, 1
c[[0, -1], :] = 1
c[:, [0, -1]] = 1
print(c)
print()

# c[1:-1, 1:-1] += 7
# print(c)

# 문제
# 앞에서 만든 2차원 배열의 양쪽 대각선을 2로 채우세요
# c[0, 0] = 2
# c[1, 1] = 2
c[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 2
c[range(5), list(reversed(range(5)))] = 2

# for i in range(5):
#     c[i, i] = 2
#     c[i, 4-i] = 2

print(c)
print()

d1 = [1, 3, 5, 6, 2]
d2 = [4, 2, 8, 1, 3]

plt.subplot(2, 2, 1)
plt.plot(range(5), d1, 'r')

# plt.figure()
plt.subplot(2, 2, 4)
plt.plot(range(5), d2, 'g')
plt.show()







