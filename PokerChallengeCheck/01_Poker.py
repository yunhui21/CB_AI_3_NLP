# ------------------------------ #
# 등수: 1
# 캐글: 1.0
# 피씨: 1.0
# ------------------------------ #

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd

# 데이터셋 로드 및 확인
train_dataset = pd.read_csv('data/train.csv') # 학습 데이터셋
test_dataset = pd.read_csv('data/test.csv') # 테스트 데이터셋
sns.countplot(train_dataset['CLASS'])

print(train_dataset['CLASS'].value_counts())

n_x_train = train_dataset.iloc[:, 1:11]
print(n_x_train)

n_y_train = train_dataset.iloc[:, 11]

# 모듈 작업
def draw_plot(history):  # 결과 시각화
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(2)
    ax[0].plot(range(1, len(acc) + 1), acc, 'b', label='accuracy')
    ax[1].plot(range(1, len(acc) + 1), val_acc, 'b', label='val_accuracy')
    ax[1].plot(range(1, len(acc) + 1), loss, 'r', label='val_loss')
    ax[0].plot(range(1, len(acc) + 1), val_loss, 'r', label='loss')
    plt.show()

def default_model_maker(config):  # 모델 생성을 위한 함수
    _model = Sequential()

    for i, cfg in config.items():
        unit = cfg.get('unit')
        activation = cfg.get('activation')
        input_shape = cfg.get('input_shape')
        kernel_initializer = cfg.get('kernel_initializer', 'glorot_uniform')
        dropout = cfg.get('dropout')

        if activation == 'mish':
            activation = 'mish'

        if i == 1:
            layer = tf.keras.layers.Dense(unit, activation=activation, input_shape=input_shape,
                                          kernel_initializer=kernel_initializer)
        else:
            if dropout:
                layer = tf.keras.layers.Dropout(0.05)
            else:
                layer = tf.keras.layers.Dense(unit, activation=activation, kernel_initializer=kernel_initializer)

        _model.add(layer)

    return _model


def calc_new_features(row):  # suit와 distance 계산
    shapes = [0, 0, 0, 0]
    for suit in row.loc[['S1', 'S2', 'S3', 'S4', 'S5']]:
        shapes[suit - 1] += 1

    if 5 in shapes:  # 플러쉬 판단을 쉽게 하기 위함
        shapes = sorted(shapes)

    _list = sorted(list(row.loc[['C1', 'C2', 'C3', 'C4', 'C5']]))
    differences = [abs(_list[i] - _list[i + 1]) for i in range(len(_list) - 1)]
    differences.append(abs(_list[len(_list) - 1] - _list[0]))
    return pd.Series(shapes + differences)

# 분류 모델 작업
BATCH_SIZE = 32
EPOCHS = 500
callbacks = [ # DNN 모델 학습 사용할 콜백 함수 선언
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
    tf.keras.callbacks.ModelCheckpoint('/', monitor='val_accuracy', save_best_only=True),
]

# 성능 개선
    # 기본 데이터셋을 기반으로 학습 시킨 결과 ACC가 만족스럽지 못함
    # Accuracy 향상을 위해 포커 특성과 관련된 features 추가
    # 포커의 카드는 순서가 중요하지 않고, Suit와 Rank가 중요
        # EX) [2, 2, 0, 1, 1] => two pair = [0, 1, 1, 2, 2] => two pair
        # EX) [1, 3, 2, 4, 5], 5 하트 => Straight Flush = [1, 2, 3, 4, 5], 5하트
        # 즉, Suit, Rank 정보를 통해 직관적으로 Case 계산 가능
n_td = train_dataset.copy() # 학습 데이터셋 복사
n_td[['a', 'b', 'c', 'd', 'ad1', 'ad2', 'ad3', 'ad4', 'ad5']] = n_td.apply(calc_new_features, axis=1) # 새로운 피쳐 계산
print(n_td) # 새로운 특성이 포함된 테이블

n_td_test = test_dataset.copy() # 테스트 데이터셋 복사
n_td_test[['a', 'b', 'c', 'd', 'ad1', 'ad2', 'ad3', 'ad4', 'ad5']] = n_td_test.apply(calc_new_features, axis=1)
print(n_td_test)

# 모델 선정
extract_cols = ['a', 'b', 'c', 'd', 'ad1', 'ad2', 'ad3', 'ad4', 'ad5'] # 추출 컬럼 정의
target_datas = n_td_test[extract_cols] # 새롭게 만들어진 특성 추출
x_train, x_test, y_train, y_test = train_test_split(n_td[extract_cols],
                                                    n_td['CLASS'],
                                                    test_size=0.3,
                                                    random_state=42,
                                                    shuffle=True) # 데이터셋 분리

# KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)

# SVM
svm = SVC()
svm.fit(x_train, y_train)
svm.score(x_test, y_test)

# Decision Tree
dcclf = DecisionTreeClassifier(max_features=9)
dcclf.fit(x_train, y_train)
dcclf.score(x_test, y_test)

# Random Forest
rfclf = RandomForestClassifier(300, max_features=9)
rfclf.fit(x_train, y_train)
rfclf.score(x_test, y_test)

# DNN
# 3 layers 모델 정의
model = default_model_maker({
    1 : {
        'unit' : 30,
        'activation' : 'relu',
        'kernel_initializer' : 'uniform',
        'input_shape' : (9, )
    },
    2 : {
        'unit' : 30,
        'activation' : 'relu',
        'kernel_initializer' : 'uniform'
    },
    3 : {
        'unit' : 10,
        'activation' : 'softmax'
    }
})

# 모델 설정
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

DNN_Y = to_categorical(n_td['CLASS']) # CLASS 정보 원핫인코딩

# 모델 학습
his = model.fit(
    n_td[extract_cols],
    DNN_Y,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_split=0.3,
    callbacks=callbacks,
    verbose=1,
)

draw_plot(his) # ACC, LOSS 시각화
loss, accuracy = model.evaluate(n_td[extract_cols], DNN_Y)

# 결과
    # KNN - 0.8393
    # SVM - 0.9087
    # DNN - 0.9996
    # DT - 1.0000
    # RF - 1.0000
# SELECT Random Forest

# 예측
pred_values = rfclf.predict(target_datas) # 테스트 데이터셋 예측
result_table = pd.concat([n_td_test, pd.Series(pred_values, name='CLASS')], axis=1) # 테스트 데이터셋과 결과 데이터셋 병합
print(result_table.head())

# 결측 데이터
print(train_dataset['CLASS'].value_counts())

    # 8번 데이터의 부재 -> 8번 (Straight Flush) 학습 불가
    # 해결 방안 -> 8번 데이터 처리

# Straight Flush Cases [모든 카드의 suit이 동일하고, 카드의 등급이 1씩 증가할 경우 => 각 카드의 값 차이가 1이면 1씩 증가]
case8 = result_table[(result_table.S1 == result_table.S2)
                   & (result_table.S2 == result_table.S3)
                   & (result_table.S3 == result_table.S4)
                   & (result_table.S4 == result_table.S5)
                   & (result_table.ad1 == 1)
                   & (result_table.ad2 == 1)
                   & (result_table.ad3 == 1)
                   & (result_table.ad4 == 1)]
# case8 # 17건의 잘못된 데이터 검출

print(len(result_table)-17*100/len(result_table))

# 8번 케이스를 처리하지 않을 경우 최대 ACC는 99.9983
for x in case8.index: # 8번 케이스 처리
  result_table.loc[x]['CLASS'] = 8
print(result_table[['Id', 'CLASS']]) # 최종 결과 테이블

result_table[['Id', 'CLASS']].to_csv('outputs/submission.csv', index=False)
