#

# 99_Poker.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection
from operator import itemgetter

# 1번 스케일링
def apply_scaling(x):
    return preprocessing.scale(x)


# 2번 원핫 벡터
def apply_onehot(x):
    # suit : 4가지 * 5 = 20
    # card : 13가지 * 5 = 65

    enc = preprocessing.LabelBinarizer()

    # binds = []
    # for i in range(x.shape[1]):
    #     binds.append(enc.fit_transform(x[:, i]))
    #     print(binds[-1].shape)
    #
    # return np.hstack(binds)

    return np.hstack([enc.fit_transform(x[: i])for i in range(x.shape[1])])


# 3번 정렬
def apply_sort_1(x):
    # print(x[0])             # [ 2. 10.  1.  4.  3. 10.  1. 10.  2. 11.]
    x = [np.reshape(i, [5, 2]) for i in x]
    # print(x[0])             # [[ 2. 10.] [ 1.  4.] [ 3. 10.] [ 1. 10.] [ 2. 11.]]
    x = [sorted(i, key=itemgetter(1, 0)) for i in x]
    # print(x[0])             # [array([1., 4.], dtype=float32), array([ 1., 10.], dtype=float32), array([ 2., 10.], dtype=float32), array([ 3., 10.], dtype=float32), array([ 2., 11.], dtype=float32)]
    x = [np.reshape(i, (-1,)) for i in x]
    # print(x[0])             # [ 1.  4.  1. 10.  2. 10.  3. 10.  2. 11.]

    return np.int32(x)

# 4번 정렬(숫자를 앞에, 무늬를 뒤에)
def apply_sort_2(x):
    cards = x[:, 1::2]
    # print(cards)          # [[10.  4. 10. 10. 11.][ 7.  4. 10.  1.  3.][13.  7. 10. 13.  4.]...
    suits = x[:, 0::2]
    # print(suits)          # [[2. 1. 3. 1. 2.][4. 4. 3. 1. 2.][2. 4. 1. 1. 3.]
    cards.sort()

    return np.hstack([cards, suits])


# 5. 피쳐추가
def apply_features(x):
    # flush 피쳐(bool)
    suits = [(r[0] == r[2] and r[0] == r[4] and r[0] == r[6] and r[0] == r[8])  for r in x]  # r에는 10개의 데이터

    # strait 피쳐(bool)
    straits = []
    for r in x:
        d = sorted(r[1::2])
        straits.append(d[0]+1 == d[3] and d[0]+2 == d[2] and d[0]+3 == d[3] and d[0]+4 == d[4])
    return np.hstack([x, np.transpose([suits, straits])])

poker = pd.read_csv('data/poker_train.csv', index_col=0)
# print(poker)

x = poker.values[:, :-1]
y = poker.values[:, -1:]

x = np.float32(x)

# print(y.shape) # (25010, 1)
# x = apply_scaling(x)
# x = apply_onehot(x)
# x = apply_sort_1(x)
x = apply_sort_2(x)
# x = apply_features(x)

# exit(-1)


# ----------------------------------- #

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.8)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train,
          epochs=100, batch_size=100, verbose=2,
          validation_data=(x_valid, y_valid))

# baseline :
# scale    :
# onehot   :
# sort 1   :
# sort 2   :
# feature  : val_loss: 0.5658 - val_acc: 0.7775