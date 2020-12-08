# ------------------------------ #
# 등수: 3
# 캐글: 0.99955
# 피씨: 0.99889
# ------------------------------ #

import tensorflow as tf
import pandas as pd
import numpy as np

print(tf.__version__)
print(pd.__version__)
# load train.csv
train_dataset = pd.read_csv("data/train.csv")

# data information
print(train_dataset.shape)
print(train_dataset.info())
# print(train_dataset.describe())
print(train_dataset.loc[0:5])

print(train_dataset['CLASS'].value_counts())
# class_num = train_dataset['CLASS'].value_counts()

print(train_dataset.head())
# load test.csv
test_dataset = pd.read_csv("data/test.csv")

# data information
print(test_dataset.shape)
print(test_dataset.info())
# print(test_dataset.describe())
print(test_dataset.loc[0:5])
print(test_dataset.head(10))

test_id = test_dataset.iloc[:, 0].values
print(test_dataset.shape[0])
print(test_id.shape)
# existed training data is imbalaced class data

# make data instance
import random

# make CLASS 9 instance
for i in range(160):
    rank_list = [10, 11, 12, 13, 1]
    random.shuffle(rank_list)
    suit = random.randint(1, 4)

    train_dataset = train_dataset.append({
        'Id': 1,
        'S1': suit, 'C1': rank_list[0],
        'S2': suit, 'C2': rank_list[1],
        'S3': suit, 'C3': rank_list[2],
        'S4': suit, 'C4': rank_list[3],
        'S5': suit, 'C5': rank_list[4],
        'CLASS': 9},
        ignore_index=True)

# print(train_dataset.isnull().values.any())
# print(train_dataset.isnull().sum().sum())

# make CLASS 8 instance
# training dataset dosen't have CLASS 8
for i in range(160):
    first_num = random.randint(1, 13)
    rank_list = []
    for j in range(5):
        rank_list.append((first_num + j - 1) % 13 + 1)
    random.shuffle(rank_list)
    suit = random.randint(1, 4)

    train_dataset = train_dataset.append({
        'Id': 1,
        'S1': suit, 'C1': rank_list[0],
        'S2': suit, 'C2': rank_list[1],
        'S3': suit, 'C3': rank_list[2],
        'S4': suit, 'C4': rank_list[3],
        'S5': suit, 'C5': rank_list[4],
        'CLASS': 8},
        ignore_index=True)

# make CLASS 5 instance
import copy

all_rank_list = list(range(1, 14))
for i in range(60):
    rank_list = random.sample(all_rank_list, 5)

    if len(set(rank_list)) == 2:  # full house or four card
        continue

    if len(rank_list) == len(set(rank_list)):  # check staright flush, royal flush
        check_list = copy.deepcopy(rank_list)
        for j in range(5):
            if check_list[j] < 5:  # in order to check case having 13, 1
                check_list[j] += 13
        if (max(check_list) - min(check_list)) == 4:
            continue
        if (max(rank_list) - min(rank_list)) == 4:
            continue

    suit = random.randint(1, 4)

    train_dataset = train_dataset.append({
        'Id': 1, 'S1': suit, 'C1': rank_list[0], 'S2': suit, 'C2': rank_list[1], 'S3': suit, 'C3': rank_list[2],
        'S4': suit, 'C4': rank_list[3], 'S5': suit, 'C5': rank_list[4], 'CLASS': 5},
        ignore_index=True)

# make CLASS 7 instance
for i in range(24):
    suit = i % 4 + 1
    rank = i % 13 + 1
    train_dataset = train_dataset.append({
        'Id': 1, 'S1': suit, 'C1': rank, 'S2': suit, 'C2': rank, 'S3': suit, 'C3': rank,
        'S4': suit, 'C4': rank, 'S5': suit, 'C5': rank, 'CLASS': 7},
        ignore_index=True)

# make CLASS 4 instance
for i in range(360):
    first_num = random.randint(1, 13)
    rank_list = []
    for j in range(5):
        rank_list.append((first_num + j - 1) % 13 + 1)
    random.shuffle(rank_list)
    suit_list = []
    for j in range(5):
        suit_list.append(random.randint(1, 4))

    if suit_list.count(suit_list[0]) == 5:
        continue
    train_dataset = train_dataset.append({
        'Id': 1, 'S1': suit_list[0], 'C1': rank_list[0], 'S2': suit_list[1], 'C2': rank_list[1], 'S3': suit_list[2],
        'C3': rank_list[2],
        'S4': suit_list[3], 'C4': rank_list[3], 'S5': suit_list[4], 'C5': rank_list[4], 'CLASS': 4},
        ignore_index=True)

# make CLASS 6 instance
all_suit_list = [1, 2, 3, 4]
all_rank_list = list(range(1, 14))
print(all_suit_list)
for i in range(80):
    rank_list = random.sample(all_rank_list, 2)
    first_rank = rank_list[0]
    rank_list.extend([rank_list[0], rank_list[0], rank_list[1]])
    random.shuffle(rank_list)

    suit_list1 = random.sample(all_suit_list, 3)
    suit_list2 = random.sample(all_suit_list, 2)
    suit_list = [0, 0, 0, 0, 0]
    j = 0
    k = 0
    for x in range(5):
        if rank_list[x] == first_rank:
            suit_list[x] = suit_list1[j]
            j += 1
        else:
            suit_list[x] = suit_list2[k]
            k += 1
    train_dataset = train_dataset.append({
        'Id': 1, 'S1': suit_list[0], 'C1': rank_list[0], 'S2': suit_list[1], 'C2': rank_list[1], 'S3': suit_list[2],
        'C3': rank_list[2],
        'S4': suit_list[3], 'C4': rank_list[3], 'S5': suit_list[4], 'C5': rank_list[4], 'CLASS': 6},
        ignore_index=True)

# train_dataset = train_dataset.fillna(0)
train_dataset = train_dataset.astype(int)
train_dataset = train_dataset.sample(frac=1)  # shuffle

print(train_dataset['CLASS'].value_counts())
df_columns = list(train_dataset.columns)

# # resampling(oversampling)
# # SMOTE
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(ratio='auto', kind='regular')

# train_x = train_dataset.iloc[:, 1:-1]
# train_y = train_dataset.iloc[:, -1]
# train_x, train_y = smote.fit_sample(train_x, train_y)
# # train_dataset = train_x + train_y
# # train_x.join(train_y)
# # train_datasetat1 = pd.concat([train_x, train_y], axis=1)
# # np.append(a, z, axis=1)
# # train_dataset = np.concatenate((train_x, train_y), axis=0)
# pd.Series(train_y).value_counts()
# train_y = train_y.reshape(-1,1)

# train_dataset = np.append(train_x, train_y, axis=1)

# train_dataset = pd.DataFrame(train_dataset, columns=df_columns[1:])
# print(train_dataset['CLASS'].value_counts()


# # use weights
# index = train_dataset['CLASS'].value_counts().index.values
# values = train_dataset['CLASS'].value_counts().values

# num = len(train_dataset['CLASS'].value_counts())
# print(values)
# print(type(values))
# class_weights = {}
# total = train_dataset.shape[0]
# for x in range(num):
#   i = index[x]

#   class_num = values[i]

#   weight = (1 / class_num)*(total)/2.0

#   # n =  chr(i+ord('0'))
#   n = i
#   class_weights[n] = weight
# print(class_weights)


# use weight1 2
# class_weights2 = {
#   0 : 1,  1 : 1,  2 : 1,  3 : 1,
#   4 : 1, 5 : 5, 6 : 5,
#   7 : 10, 8 : 10, 9 : 10
# }
# normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_columns = list(train_dataset.columns)
df_columns.pop()
df_columns.pop(0)
df_columns = [df_columns]

print(df_columns)
for df_column in df_columns:
    train_dataset[df_column] = scaler.fit_transform(train_dataset[df_column])
    test_dataset[df_column] = scaler.fit_transform(test_dataset[df_column])

# train data split
# train_x = train_dataset.iloc[:, 0:-1].values # when use smote
train_x = train_dataset.iloc[:, 1:-1].values
train_y = train_dataset.iloc[:, -1].values

# test data split
test_x = test_dataset.iloc[:, 1:].values
# test_y = test_dataset.iloc[:-1].values

# from sklearn.model_selection import train_test_split
# train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=321)

# train data split into train and test
# test data is used for model.evaluate
# from sklearn.model_selection import train_test_split
# train_x, train_test_x, train_y, train_test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=321)
# training

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(10,), activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.SGD(0.01)
# optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

checkpoint_filepath = "outputs/checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    # monitor='val_accuracy',
    monitor='accuracy',
    mode='max',
    save_best_only=True
)

hist = model.fit(train_x, train_y,
                 # validation_data = (valid_x, valid_y),
                 batch_size=32, epochs=1200,
                 # class_weight=class_weights2,
                 callbacks=[model_checkpoint_callback],
                 use_multiprocessing=True)
model.load_weights(checkpoint_filepath)  # The model weights (that are considered the best) are loaded into the model.

# model.evaluate(test_x, test_y, verbose=2, batch_size=100, use_multiprocessing=True)

prediction = model.predict(test_x)
prediction_class = tf.argmax(prediction, 1)
# visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Cost Graph")
plt.ylabel("cost")
plt.plot(hist.history['loss'])
plt.subplot(1, 2, 2)
plt.title("Accuracy Graph")
plt.ylabel('accuracy')
plt.plot(hist.history['accuracy'], 'b-', label='training accuracy')
# plt.plot(hist.history['val_accuracy'], 'r:', label='validation accuracy')
plt.legend()
plt.tight_layout()
plt.show()
# make DataFrame and store data as csv

submission = pd.DataFrame({
    "ID": test_id,
    "CLASS": prediction_class
})

submission.to_csv('outputs/submission_YuheonSong.csv', index=False)
# submission2 = pd.read_csv('submission2.csv')
# submission2.head()

# from sklearn.metrics import classification_report
# print(classification_report(test_y, prediction_class))
