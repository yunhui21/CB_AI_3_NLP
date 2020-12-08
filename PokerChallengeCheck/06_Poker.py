# ------------------------------ #
# 등수: 6
# 캐글: 0.99807
# 피씨: 0.99798
# ------------------------------ #

# Module Import
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()
import warnings
# for dirname, _, filenames in os.walk('/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')


# Train Set + preprocessing
def get_data_train():
    poker = pd.read_csv('data/train.csv', index_col=0)

    y_train = poker.values[:, -1]

    poker.drop(['CLASS'], axis=1, inplace=True)
    data = poker[['C1', 'C2', 'C3', 'C4', 'C5']]
    data.values.sort()
    poker[['C1', 'C2', 'C3', 'C4', 'C5']] = data
    poker['M1'] = poker['C5'] - poker['C4']
    poker['M2'] = poker['C4'] - poker['C3']
    poker['M3'] = poker['C3'] - poker['C2']
    poker['M4'] = poker['C2'] - poker['C1']
    x_train = poker.values
    x_train = preprocessing.scale(x_train)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_train, y_train, train_size=0.7)

    return x_train, x_test, y_train, y_test


# Test Set + preprocessing
def get_data_test():
    poker = pd.read_csv('data/test.csv', index_col=0)
    data = poker[['C1', 'C2', 'C3', 'C4', 'C5']]
    data.values.sort()
    poker[['C1', 'C2', 'C3', 'C4', 'C5']] = data
    poker['M1'] = poker['C5'] - poker['C4']
    poker['M2'] = poker['C4'] - poker['C3']
    poker['M3'] = poker['C3'] - poker['C2']
    poker['M4'] = poker['C2'] - poker['C1']

    x_test = poker.values
    x_test = preprocessing.scale(x_test)
    ids = poker.index.values

    return x_test, ids


# Show accuracy
def show_accuracy_sparse(preds):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)
    return preds_arg


# Make submission
def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')

    print('Id,CLASS', file=f)

    for i in range(len(ids)):
        result = preds[i]
        print('{},{}'.format(ids[i], result), file=f)

    f.close()


#Define Model
x_train, x_test, y_train, y_test = get_data_train()


def get_model(h1, h2, h3, h4):
    with distribution.scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(h1, kernel_initializer='he_normal',activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(h2, kernel_initializer='he_normal',activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.Dense(h3, kernel_initializer='he_normal',activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(h4, kernel_initializer='he_normal',activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(10, kernel_initializer='he_normal',activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['acc'])

    return model


# Fit model

patient = 40
epoch = 200

x_val, ids = get_data_test()

result = np.zeros([len(x_val), 10])
ensemble = 5

acc = []
j = 1

h1_list = [800, 750, 700, 650, 600, 550, 500, 450, 400, 350]
h2_list = [400, 360, 330, 300, 270, 230, 200, 170, 140, 100]
h3_list = [800, 750, 650, 700, 500, 550, 600, 400, 450, 350]
h4_list = [400, 360, 330, 300, 270, 230, 200, 170, 140, 100]

for i in range(ensemble):
    batch_size = [400, 300, 200, 100, 32]
    model_path = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch, batch_size[i])
    callbacks1 = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patient, mode='auto', verbose=2),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patient / 2, min_lr=0.0001,
                                             verbose=2, mode='auto'),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=2, save_best_only=True,
                                           mode='auto')
    ]

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size[i])
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size[i])

    model = get_model(h1_list[i], h2_list[i], h3_list[i], h4_list[i])
    history = model.fit(dataset, epochs=epoch, batch_size=batch_size[i], verbose=2,
                        validation_data=val_dataset, callbacks=callbacks1, shuffle=True)
    # history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size[i], verbose=2,
    #                     validation_data=(x_test, y_test), callbacks=callbacks1, shuffle=True)

y_pred = model.predict(x=x_val)
result += y_pred

print('-' * 30)
print(show_accuracy_sparse(result))
cls_pred = show_accuracy_sparse(result)

file_path = 'outputs/submission_jinsung4069.csv'
make_submission(file_path, ids, cls_pred)