# Day_27_01_functional.py

import tensorflow as tf
import numpy as np
# 문제
# AND 데이터셋에 대해 정확도를 계산하는 모델을 만드세요.
def and_sequential():
    data = [[0,0,0],
            [1,0,0],
            [0,1,0],
            [1,1,1]]

    data = np.int32(data)
    x = data[:, :-1]
    y = data[:, -1:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[2]))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc:', model.evaluate(x, y, verbose=0))
    print(model.predict(x))


def xor_sequential():
    data = [[0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]]

    data = np.int32(data)
    x = data[:, :-1]
    y = data[:, -1:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[2]))

    model.add(tf.keras.layers.Dense(9, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc:', model.evaluate(x, y, verbose=0))
    print(model.predict(x))


def and_functional_basic():
    data = [[0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]]

    data = np.int32(data)
    x = data[:, :-1]
    y = data[:, -1:]

    input  = tf.keras.layers.Input(shape=[2])

    # 1번 __call__의미를 이해하고 있어야 한다.
    # dense1  = tf.keras.layers.Dense(5, activation='relu')
    # output1 = dense1.__call__(input)
    # dense2  = tf.keras.layers.Dense(1, activation='sigmoid')
    # output2 = dense2.__call__(output1)

    # 2번
    # dense1  = tf.keras.layers.Dense(5, activation='relu')
    # output1 = dense1(input)
    # dense2  = tf.keras.layers.Dense(1, activation='sigmoid')
    # output2 = dense2(output1)

    # 3번
    output1  = tf.keras.layers.Dense(5, activation='relu')(input)
    output2  = tf.keras.layers.Dense(1, activation='sigmoid')(output1)

    model = tf.keras.Model(input, output2)

    # model = tf.keras.Sequential()
    # model.add()
    #
    # model.add(tf.keras.layers.Dense(9, activation='relu'))
    # model.add(tf.keras.layers.Dense(5, activation='relu'))
    # model.add()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=1000, verbose=2)
    print('acc:', model.evaluate(x, y, verbose=0))
    print(model.predict(x))


def and_functional_multi_input():
    data = [[0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]]

    data = np.int32(data)
    x1 = data[:, :1]
    x2 = data[:, 1:2]

    y = data[:, -1:]

    # 1번
    # input1 = tf.keras.layers.Input(shape=[1])
    # input2 = tf.keras.layers.Input(shape=[1])
    #
    # input  = tf.keras.layers.concatenate([input1, input2], axis=1)
    #
    # output1 = tf.keras.layers.Dense(5, activation='relu')(input)
    # output2 = tf.keras.layers.Dense(1, activation='sigmoid')(output1)
    #
    # model = tf.keras.Model([input1, input2], output2)
    # model.summary()

    # 2번
    input1  = tf.keras.layers.Input(shape=[1])
    output1 = tf.keras.layers.Dense(5, activation='relu')(input1)

    input2  = tf.keras.layers.Input(shape=[1])
    output2 = tf.keras.layers.Dense(5, activation='relu')(input2)

    concat  = tf.keras.layers.concatenate([input1, input2], axis=1)
    output3 = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

    model = tf.keras.Model([input1, input2], output3)
    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit([x1, x2], y, epochs=1000, verbose=2)
    print('acc:', model.evaluate([x1, x2], y, verbose=0))

    print(model.predict([x1,x2]))


# 문제
# andd와 xor 데이터를 한번에 처리하는 모델을 만드세요.

def and_functional_multi_inout():
    data = [[0, 0, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 1]]

    data = np.int32(data)

    x1 = data[:, 0:1]
    x2 = data[:, 1:2]
    y1 = data[:, 2:3]
    y2 = data[:, 3:4]

    # 1번
    # input1 = tf.keras.layers.Input(shape=[1])
    # input2 = tf.keras.layers.Input(shape=[1])
    # input  = tf.keras.layers.concatenate([input1, input2], axis=1)
    # output1 = tf.keras.layers.Dense(5, activation='relu')(input)
    # output2 = tf.keras.layers.Dense(1, activation='sigmoid')(output1)
    # model = tf.keras.Model([input1, input2], output2)
    # model.summary()

    # 2번
    input1  = tf.keras.layers.Input(shape=[1])
    output1 = tf.keras.layers.Dense(5, activation='relu')(input1)

    input2  = tf.keras.layers.Input(shape=[1])
    output2 = tf.keras.layers.Dense(5, activation='relu')(input2)

    concat  = tf.keras.layers.concatenate([input1, input2], axis=1)

    output3 = tf.keras.layers.Dense(1, activation='sigmoid', name = 'output3')(concat)
    output4 = tf.keras.layers.Dense(1, activation='sigmoid', name = 'output4')(concat)

    model = tf.keras.Model([input1, input2], [output3, output4])
    # model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit([x1,x2] ,[y1,y2], epochs=1000, verbose=2)
    print('acc:', model.evaluate([x1,x2],[y1,y2], verbose=0))
    print(history.history.keys())
    # dict_keys(['loss', 'dense_2_loss', 'dense_3_loss', 'dense_2_acc', 'dense_3_acc'])
    # dict_keys(['loss', 'output3_loss', 'output4_loss', 'output3_acc', 'output4_acc'])

    print(model.predict([x1,x2]))

# and_sequential()
# xor_sequential()
# and_functional_basic()
# and_functional_multi_input()
and_functional_multi_inout()