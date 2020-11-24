# Day_09_02_RnnBasic_3_helper.py
import tensorflow as tf
import numpy as np


def show_sequence_loss(targets, logits):
    y = tf.constant(targets)
    z = tf.constant(logits)

    w = tf.ones([1, len(targets[0])])

    loss = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=w)

    # 요기 tfa 설치해야 함

    sess = tf.compat.v1.Session()
    print(sess.run(loss))
    sess.close()


def sequence_loss_basic():
    # 문제
    # p1의 shape은 어떻게 됩니까?
    p1 = [[[0.2, 0.7], [0.5, 0.3], [0.1, 0.4]]]     # (1, 3, 2)
    p2 = [[[0.7, 0.2], [0.3, 0.5], [0.4, 0.1]]]
    # print(np.float32(p1).shape)

    # sparse: [[1, 1, 1]]
    # dense : [[[0, 1], [0, 1], [0, 1]]]
    show_sequence_loss([[1, 1, 1]], p1)
    show_sequence_loss([[0, 0, 0]], p2)

    # 문제
    # 아래 코드가 에러나지 않도록 수정하세요
    # show_sequence_loss([[1, 1, 1, 1]], p1)
    show_sequence_loss([[1, 1, 1, 1]],
                       [[[0.2, 0.7], [0.5, 0.3], [0.1, 0.4], [0.3, 0.2]]])

    # 문제
    # 아래 코드가 에러나지 않도록 수정하세요
    # show_sequence_loss([[2, 2, 2]], p1)
    show_sequence_loss([[2, 2, 2]],
                       [[[0.2, 0.7, 0.1], [0.5, 0.3, 0.2], [0.1, 0.4, 0.5]]])
    # sparse: [[2, 2, 2]]
    # dense : [[[0, 0, 1], [0, 0, 1], [0, 0, 1]]]


def show_matmul():
    a = np.array([[1, 2, 3], [4, 5, 6]])    # (2, 3)
    b = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)

    # print(a.shape, b.shape)
    print(np.dot(a, b).shape)   # (2, 3) @ (3, 2) = (2, 2)
    print(np.dot(b, a).shape)   # (3, 2) @ (2, 3) = (3, 3)
    print()

    print(tf.matmul(a, b).shape)
    print(tf.matmul(b, a).shape)
    print('----------------------------------')

    aa = a[np.newaxis]          # (1, 2, 3)
    bb = b[np.newaxis]          # (1, 3, 2)

    # print(aa.shape, bb.shape)
    print(np.dot(aa, bb).shape)     # (1, 2, 1, 2)
    print(tf.matmul(aa, bb).shape)  # (1, 2, 2)

    # 문제
    # 행렬 곱셈이 성립하는 3차원 배열 2개를 만드세요
    aaa = np.arange(24).reshape(3, 2, 4)
    bbb = np.arange(24).reshape(3, 4, 2)
    print(tf.matmul(aaa, bbb).shape)


# sequence_loss_basic()
show_matmul()

# [                 # 1
#     [             # 3
#         [         # 2
#             0.2,
#             0.7
#         ],
#         [
#             0.5,
#             0.3
#         ],
#         [
#             0.1,
#             0.4
#         ]
#     ]
# ]



