# Day_08_01_RnnBasic_1.py
import tensorflow as tf
import csv
import numpy as np
# 리니어 리그레션, 멀티플 리그레션
# 로지스틱 리그레션, 소프트맥스 리그레션

def review_softmax():
    # 문제
    # iris 데이터 파일을 읽어서 x와 y로 분리해서 반환하는 함수를 만드세요.
    np.set_printoptions(linewidth=1000)
    def get_data():
        onehot = {'setosa':[1, 0, 0],'versicolor':[0, 1, 0],'virginica':[0, 0, 1]}

        f = open('data/iris(150).csv', 'r', encoding='utf-8')
        f.readline()

        x, y = [], []
        for row in csv.reader(f):
            print(row)
            x.append([float(v) for v in row[1:-1]])
            y.append(onehot[row[-1]])

        f.close()

        print(*x[:3], sep='\n')
        print(*y[:3], sep='\n')
        return x, y

    x, y = get_data()

    w = tf.Variable(tf.random.uniform([4, 3]))
    b = tf.Variable(tf.random.uniform([3]))

    # y = wx + b
    # y = w1 * x1 + w2 * x2 + w3 * x3 + b
    # (150, 3) = (150, 4) @ (4, 3 )
    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    # 정답까지의 거리 계산
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = z, labels= y)
    loss = tf.reduce_min(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    print('-'*30)

    preds = sess.run(hx)
    print(preds)
    print(preds.shape) # (150, 3)

    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg)

    y_arg = np.argmax(y, axis=1)
    print(y_arg)

    equals = (preds_arg == y_arg)
    print(equals)

    print('acc :', np.mean(equals))
    sess.colse()

# char rnn: word안에 포함된 char를 토큰으로 만들어서 진행
# word rnn: sent안에 포함된 word를 토믄으로 만들어서 진행

# grove
# g -> r
# r -> 0
# o -> v
# v -> e
# e -> ?    # error

# 오늘 그로브 카페에 가서 공부했다.
# 오늘 -> 그로브
# 그로브 -> 카페에

# 문제
# tensor 단어를 갖고 x와 y를 만들어서 소프트맥스 리그레션 모델을 구성하세요.

def rnn_basic():
    # tensor(t: 1 0 0 0 0 0, e: 0 1 0 0 0 0)
    # enorst(t: 0 0 0 0 0 1, e: 1 0 0 0 0 0)

    # x: tenso
    # y: ensor#
    x = [
        [0, 0, 0, 0, 0, 1],  # 5, t
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
    ]
    y = [
        [1, 0, 0, 0, 0, 0],  # 0, e
        [0, 1, 0, 0, 0, 0],  # 1, n
        [0, 0, 0, 0, 1, 0],  # 4, s
        [0, 0, 1, 0, 0, 0],  # 2, o
        [0, 0, 0, 1, 0, 0],  # 3, r
    ]
    x = np.float32(x)

    w = tf.Variable(tf.random.uniform([6, 6]))
    b = tf.Variable(tf.random.uniform([6]))

    # (5, 6) = (5, 6) @ (6, 6)
    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    # 정답까지의 거리 계산
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits = z, labels= y)
    loss = tf.reduce_min(loss_i)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

    print('-'*30)

    preds = sess.run(hx)
    preds_arg = np.argmax(preds, axis=1)

    print(preds_arg)

    word = 'enorst'
    print([i for i in preds_arg])
    print([word[i] for i in preds_arg])

    word = np.array(list('enorst'))
    print(word[preds_arg])
    sess.close()

rnn_basic()