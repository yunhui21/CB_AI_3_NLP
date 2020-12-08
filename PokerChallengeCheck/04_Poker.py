# ------------------------------ #
# 등수: 4
# 캐글: 0.99915
# 피씨: 0.99909
# ------------------------------ #

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
import os

MODEL_SAVE_FOLDER_PATH = './model/'
card = pd.read_csv('data/train.csv', index_col=0)


x = card.values[:, :-1]
y = card.values[:, -1]

scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(y)

# x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.9)

n_classes = len(enc.classes_)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:03d}_{val_loss:.4f}.h5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                monitor='loss',
                                                save_best_only=True)

model.fit(x, y,
          epochs=200, batch_size=15, verbose=2,
          validation_split=0.002,
          callbacks=[checkpoint])

print('acc :', model.evaluate(x, y, verbose=0))

# exit(-1)
# =================================================================
card_test = pd.read_csv('data/test.csv',
                        index_col=0)

x_test = scaler.transform(card_test)
preds = model.predict(x_test)
df_test = pd.DataFrame(card_test.index)
preds_arg = np.argmax(preds, axis=1)
df_test['CLASS'] = enc.classes_[preds_arg]
df_test.to_csv('outputs/submission_ilbonge.csv', index=False)