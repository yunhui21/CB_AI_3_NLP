#Day_31_01 SaveModel
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection

# 문제
# 전복 데이터를 80%로 학습하고 20%에 대해 정확도를 구하세요.
# (3개의 클래스로 재구성: ring classes 1-8, 9 and 10, and 11 on)

def get_abalone():
    names = ('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
    abalone = pd.read_csv('data/abalone.data', header=None, names=names)
    # print(abalone)

    # y = []
    # for r in abalone.rings:
    #     if   r <= 8 : y.append(0)
    #     elif r <= 10: y.append(1)
    #     else        : y.append(2)
    # print(y[:10])   # [2, 0, 1, 1, 0, 0, 2, 2, 1, 2]

    #                   1                    8     10
    # categories = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    categories = [-1] + [0] * 8 + [1] * 2 + [2] * 19
    categories = np.int32(categories)

    y = categories[abalone.rings]
    # print(y[:10])     # [2 0 1 1 0 0 2 2 1 2]

    # x = abalone.values[:, 1:-1]
    # print(x.shape, x.dtype) # (4177, 7) object
    # x = np.float32(x)

    x = abalone.drop(['sex', 'rings'], axis=1).values
    # print(x.shape, x.dtype) # (4177, 7) float64

    sex = preprocessing.LabelBinarizer().fit_transform(abalone.sex)
    # print(sex[:3])  # [[0 0 1] [0 0 1] [1 0 0]]

    # x = np.concatenate([sex, x], axis=1)
    x = np.hstack([sex, x])
    x = preprocessing.scale(x)
    # print(x.shape)       # (4177, 10)

    return model_selection.train_test_split(x, y, train_size=0.8)

def build_model(n_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(60, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss = tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    return model

def save_model_1():
    x_train, x_test, y_train, y_test = get_abalone()
    model = build_model(n_classes = len(set(y_train)))

    model.fit(x_train, y_train, epochs=10, verbose=0)
    print('acc:', model.evaluate(x_test, y_test))

    # acc: [0.7922932505607605, 0.5944976210594177]  - non-scaling
    # acc: [0.7185429930686951, 0.6578947305679321]  - scaling

    # model.save('model_abalone/keras.h5') # keras 버전 편하지만..정보가 적다.
    # model.save('model_abalone/keras.h5', save_format='h5') # keras 버전 편하지만..정보가 적다.
    #
    # model.save('model_abalone') # tensorflow 버전 꼭 사용. 실전에 꼭 사용.
    # model.save('model_abalone', save_format='tf') # tensorflow 버전 꼭 사용. 실전에 꼭 사용.

    json = model.to_json()
    print(json)
    # {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 10], "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 60, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.4.0", "backend": "tensorflow"}
    # 아키텍쳐 전달은 어렵지만 외부전달에는 가장 좋다.

    model_json = tf.keras.models.model_from_json(json)
    print(model.to_yaml())      # PyYAML
    # model.save_weights()


def save_model_2_bug():
    x_train, x_test, y_train, y_test = get_abalone()
    model = build_model(n_classes=len(set(y_train)))

    model.fit(x_train, y_train, epochs=10, verbose=0)
    print('acc 1:', model.evaluate(x_test, y_test, verbose=0))
    model_path = 'model_abalone/bug.h5'
    model.save(model_path)
    # acc 1: [0.7180963754653931, 0.6507176756858826]

    save_model = tf.keras.models.load_model(model_path)
    print('acc 2:', save_model.evaluate(x_test, y_test, verbose=0))
    # acc 2: [0.7180963754653931, 0.3229665160179138]

    preds = save_model.predict(x_test)
    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg[:10])
    print('acc 3:', np.mean(preds_arg == y_test))

    # compile 함수를 호출하지 않으면 evaluate 함수는 비정상으로 동작한다.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])
    print('acc 4:', save_model.evaluate(x_test, y_test, verbose=0))


# save_model_1()
save_model_2_bug()