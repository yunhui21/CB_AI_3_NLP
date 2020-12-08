# ------------------------------ #
# 등수: 12
# 캐글: 0.95633
# 피씨: 0.84130
# ------------------------------ #

# 1. 데이터 로드
import pandas as pd
import numpy as np

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
X_train = train_df.loc[:, train_df.columns != 'CLASS']
Y_train = train_df['CLASS']

X_test = test_df
Y_train.groupby(Y_train).size()


# 2. 자료 전처리
def preprocess_data_1(data:pd.DataFrame):
    df = data.copy()
    dfc = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    dfc.values.sort()
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = dfc
    df = df[['C1', 'S1', 'C2', 'S2', 'C3', 'S3', 'C4', 'S4', 'C5', 'S5', 'CLASS']]
    return df


def preprocess_data_2(data:pd.DataFrame):
    df = data.copy()
    dfc = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    dfc.values.sort()
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = dfc
    # df = df[['C1', 'C2', 'C3', 'C4', 'C5', 'S1', 'S2', 'S3', 'S4', 'S5']]
    return df


X_train_pre = preprocess_data_1(train_df)
X_test_pre = preprocess_data_2(test_df)

X_train = X_train_pre.loc[:, X_train_pre.columns != 'CLASS']
X_test = X_test_pre.loc[:, X_test_pre.columns != 'CLASS']

# CLASS one-hot encoring
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(train_df['CLASS'])
print(X_train_pre.head())
print(X_train_pre.shape)
print(X_train.shape)
print(X_test_pre.head())
print(Y_train.shape)

# 3. 모델링
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
model = Sequential()
model.add(Dense(24, activation='relu', input_dim=10,))
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='outputs/checkpoint', monitor='val_accuracy', save_best_only=True)
]

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(X_train, Y_train, batch_size = 100,
               epochs = 100, validation_split=0.2,
               callbacks = callbacks, shuffle=True
)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(model.evaluate(X_train, Y_train))
prediction = model.predict(X_test)
print(prediction)

# 4. 결과 저장
pred_result = np.argmax(prediction, axis=1)
submission = pd.read_csv("data/sample_submission.csv")
submission['CLASS'] = pred_result
submission.to_csv('outputs/submission_myTipp.csv', index=False)
