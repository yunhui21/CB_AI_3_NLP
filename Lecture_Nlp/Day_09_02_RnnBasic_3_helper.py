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



