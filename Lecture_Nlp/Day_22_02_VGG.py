# Day_22_02_VGG.py
import tensorflow as tf

# 문제
# slim 라이브러리에 있는 vgg 파일을 활용해서 케라스 vgg 모델을 구현하세요
# (compile, fit, evaluate 함수는 호출하지 않습니다)
# 결과는 summary 함수를 통해 shape으로 확인합니다.

# 문제
# vgg16 모델에서 마지막에 있는 dense 레이어를 conv2d 레이어로 교체하세요
# (slim 라이브러리 참고)


def model_vgg_dense():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[224, 224, 3]))

    model.add(tf.keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(1000, activation='softmax'))

    model.summary()


def model_vgg_conv2d():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[224, 224, 3]))

    model.add(tf.keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(512, [3, 3], 1, 'same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2], 2, 'same'))

    model.add(tf.keras.layers.Conv2D(4096, [7, 7], 1, 'valid', activation='relu'))
    model.add(tf.keras.layers.Conv2D(4096, [1, 1], 1, activation='relu'))
    model.add(tf.keras.layers.Conv2D(1000, [1, 1], 1, activation='softmax'))

    model.summary()


# model_vgg_dense()
model_vgg_conv2d()

