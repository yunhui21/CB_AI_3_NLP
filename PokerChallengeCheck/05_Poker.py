# ------------------------------ #
# 등수: 5
# 캐글: 0.99809
# 피씨: 0.99207
# ------------------------------ #

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection

poker = pd.read_csv('data/train.csv',
                    index_col=0)

x = poker.values[:, :-1]
y = poker.values[:, -1]
print(x.shape, y.shape)
enc = preprocessing.LabelBinarizer()
y = enc.fit_transform(y)

x = np.float32(x)
scaler = preprocessing.StandardScaler()

x = scaler.fit_transform(x)
print(x.shape, y.shape)

print(x.shape, y.shape)
# x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.8)

n_classes = len(enc.classes_)  # class의 갯수를 알기 위해서
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),  # default로 0.01
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['acc'])

# model.fit(x_train, y_train,
#           epochs=100, batch_size=32, verbose=2,
#           validation_data=(x_valid, y_valid))
model.fit(x, y,
              epochs=200, batch_size=10, verbose=2,
              validation_split=0.05)
# print('acc :', model.evaluate(x_valid, y_valid, verbose=0))
print(model.evaluate(x,y,verbose=0))

# ---------------- #

poker_test = pd.read_csv('data/test.csv',
                         index_col=0)

x_test = scaler.transform(poker_test)

preds = model.predict(x_test)

df_test = pd.DataFrame(poker_test.index)

preds_arg = np.argmax(preds, axis=1)
df_test['CLASS'] = enc.classes_[preds_arg]

df_test.to_csv('outputs/submission_choyoonhee.csv',index=False)
