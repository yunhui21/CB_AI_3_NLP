#

# 99_Poker.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection
from operator import itemgetter

poker = pd.read_csv('data/train.csv', index_col=0)
# print(poker)

x = poker.values[:, :-1]
y = poker.values[:, -1:]

x = np.float32(x)
# print(y.shape) # (25010, 1)

# 스케일링은 엄청난 효과 발생
# 2
# x = preprocessing.scale(x)

# # 원핫 벡터 변환하면 엄청난 효과 발생
# enc = preprocessing.LabelBinarizer()

# # 3
# # x_onehot = []
# # for i in range(x.shape[1]):
# #     xx = enc.fit_transform(x[:, i])
# #     x_onehot.append(xx)
# x_onehot=[enc.fit_transform(x[: i])for i in range(x.shape[1])]
# x = np.hstack(x_onehot)
# print(x.shape)

# 4 정렬
# # x : (25010, 10)
# # x = [[(xx[i], xx[i+1]) for i in range(0, 10, 2)] for xx in x]
# x = [np.reshape(i, [5, 2]) for i in x]
# # x = [sorted(xx, key=lambda t: t[1]) for xx in x]
# x = [sorted(xx, key=itemgetter(1)) for xx in x]
#
# x = [np.reshape(xx, [-1,]) for xx in x]
# x = np.int32(x)
# # print(x[:3])
# # exit(-1)

# 5 정렬
# cards = x[:, 1::2]
# cards.sort()
# suits = x[:, ::2]
# x = np.hstack([cards, suits])
# print(cards[:5])

# 피쳐추가
# straits = []
# for r in x:
#     d = sorted(r[1::2])
#     s = ( d[0]+1 == d[1] and d[0]+2 == d[2] and d[0]+3 == d[3] and d[0]+4 == d[4] )
#     straits.append(s)
#
# suits = [(r[0] == r[2] and r[0] == r[4] and r[0] == r[6] and r[0] == r[8]) for r in x]
#
# x = np.hstack([x, np.transpose([suits, straits])]) #
# print(x.shape) # (25010, 12)
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