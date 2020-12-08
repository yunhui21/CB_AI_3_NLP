# ------------------------------ #
# 등수: 7
# 캐글: 0.99797
# 피씨: error
# ------------------------------ #

import tensorflow as tf
import pandas as pd

test_x = pd.read_csv('data/test.csv')
train_x = pd.read_csv('data/train.csv')
print(train_x.head())
# import seaborn as sns
import matplotlib.pyplot as plt

test_y = pd.read_csv('data/sample_submission.csv')
test_y = test_y['CLASS']
train_y = train_x['CLASS']
del train_x['CLASS']
# from keras.utils import np_utils
# from tensorflow.keras.utils import np_utils

print(train_y.shape)
print(train_x.shape)
print(test_y.shape)
print(test_x.shape)
# train_y = np_utils.to_categorical(train_y)
# test_y = np_utils.to_categorical(test_y)
train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

print(train_y.shape)
print(train_x.shape)
print(test_y.shape)
print(test_x.shape)


def count_num(list):
    res = []
    for i in range(5):
        if list[i] == -1:
            continue

        c = list.count(list[i])
        for i2 in range(5):
            if i2 == i:
                continue
            if list[i2] == list[i]:
                list[i2] = -1
        res.append(c)
    res.sort(reverse=True)
    while (len(res) < 5):
        res.append(0)
    return res


def count_duplicate(df):
    df['D1'] = 0
    df['D2'] = 0
    df['D3'] = 0
    df['D4'] = 0
    df['D5'] = 0
    for Index in df.index:
        tmp = []
        for i in range(1, 6):
            tmp.append(df.loc[Index]['C' + str(i)])
        res = count_num(tmp)
        for i in range(1, 6):
            df.loc[Index]['D' + str(i)] = res[i - 1]
    return df


def count_sequential(df):
    df['R'] = 0
    df['S'] = 0
    for Index in df.index:
        list = []
        for i in range(1, 6):
            list.append(df.loc[Index]['C' + str(i)])
        list.sort()
        if list == [1, 10, 11, 12, 13]:
            df.loc[Index]["R"] = 1
            df.loc[Index]["S"] = 1
        else:
            if (list[0] == (list[1] - 1)) and (list[0] == (list[2] - 2)) and (list[0] == (list[3] - 3)) and (
                    list[0] == (list[4] - 4)):
                df.loc[Index]["S"] = 1
    return df


def count_suit(df):
    for Index in df.index:
        df['F'] = 0
        tmp = []
        # c = 1
        for i in range(1, 6):
            tmp.append(df.loc[Index]["S" + str(i)])
        if (tmp[0] == tmp[1]) and (tmp[1] == tmp[2]) and (tmp[2] == tmp[3]) and (tmp[3] == tmp[4]):
            df.loc[Index]['F'] = 1
    return df


train_x = count_suit(train_x)
train_x = count_duplicate(train_x)

train_x = count_sequential(train_x)
print(train_x)

del train_x['Id']
del train_x['S1']
del train_x['S2']
del train_x['S3']
del train_x['S4']
del train_x['S5']
del train_x['C1']
del train_x['C2']
del train_x['C3']
del train_x['C4']
del train_x['C5']
print(train_x)

del train_x['D3']
del train_x['D4']
del train_x['D5']
print(train_x)

test_x = count_duplicate(test_x)
test_x = count_sequential(test_x)
test_x = count_suit(test_x)

del test_x['S1']
del test_x['S2']
del test_x['S3']
del test_x['S4']
del test_x['S5']
del test_x['C1']
del test_x['C2']
del test_x['C3']
del test_x['C4']
del test_x['C5']
del test_x['D3']
del test_x['D4']
del test_x['D5']

test_x.to_csv('test_x.csv', index=False)
train_x.to_csv('train_x.csv', index=False)

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(len(train_x.keys()),)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

del train_x['Id']
del train_x['S1']
print(train_x.shape)
print(train_y.shape)
print(train_x)

history = model.fit(x=train_x, y=train_y, batch_size=16, epochs=40, validation_split=0.2)

del test_x['Id']
print(test_x.shape)
print(train_x.shape)
print(test_x)

test_x = test_x[['F', 'D1', 'D2', 'R', 'S']]
print(test_y.shape)
print(train_y.shape)

prob_pred = model.predict(test_x)
prob_label = prob_pred.argmax(axis=-1)
test_xa = pd.read_csv('data/test.csv')

submission = pd.DataFrame({
    'Id': test_xa['Id'],
    'CLASS': prob_label
})

submission.to_csv('outputs/submission_vcho1958.csv', index=False)
