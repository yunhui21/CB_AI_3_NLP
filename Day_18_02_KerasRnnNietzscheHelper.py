# Day_18_02_KerasRnnNietzscheHelper.py
import tensorflow as tf
import numpy as np
import collections

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
def softmax_1(dist):
    return dist / np.sum(dist)


def softmax_2(dist):
    dist = np.exp(dist)
    return dist / np.sum(dist)

def temperature_1(dist, t):
    dist = np.log(dist) / t
    dist = np.exp(dist)
    return dist / np.sum(dist)

def temperature_2(dist, t):
    dist = dist / t
    dist = np.exp(dist)
    return dist / np.sum(dist)

dist = [2.0, 1.0, 0.1]

print(softmax_1(dist)) # [0.65 0.32 0.03]
print(softmax_2(dist)) # [0.66 0.24 0.10]
print()

for t in np.linspace(0.1, 1.0, 10 ):
    print(temperature_1(dist, t))
print()
for t in np.linspace(0.1, 1.0, 10 ):
    print(temperature_2(dist, t))
print()

d1 = softmax_1(dist)
d2 = np.cumsum(d1)
print(d1)           # [0.65 0.32 0.03]
print(d2)           # [0.65 0.97 1.00]

print(np.searchsorted([0.5, 1.0, 2.0], 0.1))    # 0
print(np.searchsorted([0.5, 1.0, 2.0], 0.7))    # 1
print(np.searchsorted([0.5, 1.0, 2.0], 1.5))    # 2
print(np.searchsorted([0.5, 1.0, 2.0], 3.0))    # 3

print(np.random.rand(1))                        # [0.11]
print(np.searchsorted([0.5, 1.0, 2.0],
                      np.random.rand(1)))       # [0]
print()
indices = [np.searchsorted(d2, np.random.rand(1)[0]) for _ in range(100)]
print(indices)
print(collections.Counter(indices))     # Counter({0: 66, 1: 32, 2: 2})
# [0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 1, 2, 0]