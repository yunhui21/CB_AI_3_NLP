# ------------------------------ #
# 등수: 10
# 캐글: 0.99536
# 피씨: 0.99
# ------------------------------ #

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt

def get_train():
    train = pd.read_csv('data/train.csv', index_col=0, header=0)
    print(train.describe())

    y = train['CLASS'].values
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y)

    train.drop(['CLASS'], axis=1, inplace=True)

    x = train.values# + train.values # 데이터량 2배

    return np.float32(x), y

def get_test():
    test = pd.read_csv('data/test.csv', header=0)

    ids = test.Id.values
    test.drop('Id', axis=1, inplace=True)

    x = test.values

    return np.float32(x), ids


def model_poker(x_train, x_test, y_train, test_ids):
    # RNN은 시간데이터가 없고 CNN은 이미지데이터가 없어 사용할 수 없다.
    # 따라서 멀티레이어-keras 를 사용해서 풀어야 한다. (softmax-regression 사용)
    # test데이터에 정답이 나와있지 않으므로, train데이터를 7대3정도로 나눠서 7학습, 3검증 으로 테스트해본다.
    # 정확도가 90이상 나온다면 submission해보기

    # 2차원만 넣는 코드이다. mnist도 3차원이라 2차원인 (60000, 28x28=784)로 변환한다.
    model = tf.keras.Sequential()
    # 레이어 전달 ex : 10 -> 784 -> 256 -> 256 -> 10
    # 포커에 대해서 : 피처가 너무 적으면 위처럼 증폭해서 사용하면 성능이 좋아질 수 있다
    # Dense레이어에 들어가는 숫자는 다음 레이어에 전달할 출력 값
    model.add(tf.keras.layers.Input(10))
    # models.add(tf.keras.layers.Dense(1024, activation='relu'))
    # models.add(tf.keras.layers.Dropout(0.7))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3)) # Dropout은 100% 동작하지 않을수있지만, epoch를 늘려 시간적 여유를 두고 사용해야한다.
    model.add(tf.keras.layers.Dense(384, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3)) # 오버피팅이 뒤로 밀리는 효과가 있기 때문에 활용하는 것이 좋다.
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # softmax나 sigmoid는 무조건 마지막한번만 사용. 나머지는 relu 사용

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['acc'])
    # metrics=['accuracy', 'sparse_categorical_crossentropy'])
    # 원핫 벡터가 아니므로 sparse_categorical_crossentropy 사용

    model.summary()
    history = model.fit(x_train, y_train, epochs=2000, verbose=2,
    batch_size=100)#, validation_data=(x_test, y_test)) # models save, load로 시간을 단축할 수 있다.
    # print(models.evaluate(x_test, y_test, verbose=2))

    model.save('outputs/models/model_poker_2000_wj.h5')

    # plot_history([
    #('loss', history, 'loss'),
    #('acc', history, 'acc')
    # ])

    # plt.plot(range(len(history.history['loss']))[:700], history.history['loss'][:700])
    # plt.plot(range(len(history.history['acc'])[:700]), history.history['acc'][:700])
    # plt.xlabel(history.epoch)
    # # plt.show()
    # # return

    preds = model.predict(x_test)
    preds_arg = np.argmax(preds, axis=1)

    # y_test = y_test.reshape(-1, 1)
    # print(preds.shape, y_test.shape)
    #
    # show_accuracy_sparse(preds, y_test)
    make_submission(preds_arg, test_ids)

def plot_history(histories):
    plt.figure(figsize=(64,40))

    for name, history, key in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()

def show_accuracy(preds, labels):
    preds = preds.reshape(-1)
    bools = preds > 0.5
    y_bools = labels.reshape(-1)
    print('acc :', np.mean(bools == y_bools))

def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)
    print(preds_arg.shape, labels.shape)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))

def make_submission(preds, test_ids):
    files = os.listdir('outputs/submission/')
    rng = len(files) if len(files) >= 1 else 1

    filename = ''
    for f in range(1, rng+2):
        sample_filename = 'my_submission_{}.csv'.format(f)
        if not sample_filename in files:
            filename = sample_filename
            break

    file = open('outputs/submission/{}'.format(filename), 'w+')
    print('Id,CLASS', end='\n', file=file)
    for i in range(len(preds)):
        print('{},{}'.format(test_ids[i], preds[i]), end='\n', file=file)


x_train, y_train = get_train() # (25010, 10), (25010,)
x_test, test_ids = get_test() # (1000000, 10), (1000000,)

# x, y = get_train()
# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)

scaler = preprocessing.StandardScaler()
# scaler = preprocessing.MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model_poker(x_train, x_test, y_train, y_test)
model_poker(x_train, x_test, y_train, test_ids)