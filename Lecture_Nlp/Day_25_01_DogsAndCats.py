# ------------------------------------------------------------------------------------ #
# 수업 진행
#
# 1. 앞에서 진행했던 '개와 고양이'에 이어서 진행
#    기본 모델부터 전이학습까지 4가지 모델 소개
# 2. 모델 구성에 들어가기 전에 show_history 함수 등을 비롯한 유틸리티 함수 입력
#    show_history_ema 함수는 기본 그래프 보고 난 후에 구현
#    처음부터 load_model 함수를 만들 필요는 없다 (validation 결과가 있기 때문에)
#    모델을 저장한 후에 일괄적으로 봐도 전혀 문제없다 (3번은 모델 형식이 달라서 제외)
# 3. 지수이동평균 함수는 설명하지 않고 넘어갔다
#    시간 부족하다면 사용한다
# 4. 이번 수업에서 참고했던 원본 코드
#    keras_cnn_02_cats_and_dogs_2.py
#
# *. 26일차 수업은 파일이 없다
#    이번 수업 나머지와 경진대회 설명하고 코드 구현했다
# ------------------------------------------------------------------------------------ #

# Day_25_01_DogsAndCats.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 문제
# load_model 함수를 구현하세요


def get_model_name(version):
    filename = 'dogcat_small_{}.h5'.format(version)
    return os.path.join('dogs_and_cats_model', filename)


def get_history_name(version):
    filename = 'dogcat_small_{}.history'.format(version)
    return os.path.join('dogs_and_cats_model', filename)


def save_history(history, version):
    with open(get_history_name(version), 'wb') as f:
        pickle.dump(history.history, f)


def load_history(version):
    with open(get_history_name(version), 'rb') as f:
        history = pickle.load(f)
        show_history(history, version)


def show_history(history, version):
    # epochs = np.aragne(len(history['loss']))

    plt.subplot(1, 2, 1)
    plt.plot(history['val_loss'], 'g', label='valid')
    plt.plot(history['loss'], 'r', label='train')
    plt.title('loss {}'.format(version))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], 'g', label='valid')
    plt.plot(history['acc'], 'r', label='train')
    plt.title('acc {}'.format(version))
    plt.legend()

    plt.show()


# 지수 이동 평균
def show_history_ema(history, version):
    def get_ema(points, factor=0.8):
        smoothed = [points[0]]
        for pt in points[1:]:
            prev = smoothed[-1]
            smoothed.append(prev * factor + pt * (1 - factor))
        return smoothed

    loss1 = get_ema(history['val_loss'])
    loss2 = get_ema(history['loss'])

    acc1 = get_ema(history['val_acc'])
    acc2 = get_ema(history['acc'])

    epochs = np.arange(len(history['loss']))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss1, 'r', label='valid')
    plt.plot(epochs, loss2, 'g', label='train')
    plt.title('loss {}'.format(version))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc1, 'r', label='valid')
    plt.plot(epochs, acc2, 'g', label='train')
    plt.title('accuracy {}'.format(version))
    plt.legend()

    plt.show()


def load_model(version):
    model = tf.keras.models.load_model(get_model_name(version))
    model.summary()

    test_gen = ImageDataGenerator(
        rescale=1/255,
    )

    batch_size = 1000
    flow = test_gen.flow_from_directory(
        'dogs_and_cats/small/test',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )

    x, y = flow.next()
    print(x.shape, y.shape)     # (1000, 150, 150, 3) (1000,)

    print('acc :', model.evaluate(x, y, verbose=0))



def model_1_baseline():
    data_gen = ImageDataGenerator(rescale=1/255)

    batch_size = 32
    train_flow = data_gen.flow_from_directory(
        'dogs_and_cats/small/train',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    valid_flow = data_gen.flow_from_directory(
        'dogs_and_cats/small/validation',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[150, 150, 3]))

    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit_generator(
        train_flow,
        epochs=10,
        steps_per_epoch=2000 // batch_size,
        validation_data=valid_flow,
        validation_steps=batch_size,
        # verbose=2
    )
    model.save(get_model_name(version=1))
    save_history(history, 1)


def model_2_augmentation():
    train_gen = ImageDataGenerator(
        rescale=1/255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )
    valid_gen = ImageDataGenerator(
        rescale=1/255,
    )

    batch_size = 32
    train_flow = train_gen.flow_from_directory(
        'dogs_and_cats/small/train',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    valid_flow = valid_gen.flow_from_directory(
        'dogs_and_cats/small/validation',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[150, 150, 3]))

    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D([2, 2]))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit_generator(
        train_flow,
        epochs=10,
        steps_per_epoch=2000 // batch_size,
        validation_data=valid_flow,
        validation_steps=batch_size,
        # verbose=2
    )
    model.save(get_model_name(version=2))
    save_history(history, 2)


def model_3_pretrained():
    def extract_features(conv_base, data_gen, directory, sample_count, batch_size):
        x = np.zeros([sample_count, 4, 4, 512])
        y = np.zeros([sample_count])

        flow = data_gen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary'
        )

        for ii, (xx, yy) in enumerate(flow):
            n1 = ii * batch_size
            n2 = n1 + batch_size

            # 자투리가 남을 수 있지만, 처리하지 않고 탈출
            if n2 > sample_count:
                break

            x[n1:n2] = conv_base.predict(xx)
            y[n1:n2] = yy

        return x.reshape(-1, 4 * 4 * 512), y

    conv_base = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=[150, 150, 3],
    )
    conv_base.summary()

    batch_size = 32
    data_gen = ImageDataGenerator(rescale=1/255)

    x_train, y_train = extract_features(
        conv_base, data_gen, 'dogs_and_cats/small/train', 2000, batch_size)
    x_valid, y_valid = extract_features(
        conv_base, data_gen, 'dogs_and_cats/small/validation', 1000, batch_size)
    x_test, y_test = extract_features(
        conv_base, data_gen, 'dogs_and_cats/small/test', 1000, batch_size)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[4 * 4 * 512]))

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit(
        x_train, y_train,
        epochs=10, batch_size=batch_size,
        validation_data=(x_valid, y_valid),
        # verbose=2
    )
    model.save(get_model_name(version=3))
    save_history(history, 3)

    print('acc :', model.evaluate(x_test, y_test))


def model_4_pretrained_augmentation():
    train_gen = ImageDataGenerator(
        rescale=1/255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )
    valid_gen = ImageDataGenerator(
        rescale=1/255,
    )

    batch_size = 32
    train_flow = train_gen.flow_from_directory(
        'dogs_and_cats/small/train',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    valid_flow = valid_gen.flow_from_directory(
        'dogs_and_cats/small/validation',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )

    conv_base = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=[150, 150, 3],
    )
    conv_base.trainable = False

    # for layer in conv_base.layers:
    #     print(layer.name)
    #
    #     if 'block5' in layer.name:
    #         layer.trainable = True

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[150, 150, 3]))
    model.add(conv_base)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit_generator(
        train_flow,
        epochs=10,
        steps_per_epoch=2000 // batch_size,
        validation_data=valid_flow,
        validation_steps=batch_size,
        # verbose=2
    )
    model.save(get_model_name(version=2))
    save_history(history, 2)


# 시간을 측정하지 않아도 epoch마다 나오는 소요시간으로 대략 알 수 있다
start = time.time()

# model_1_baseline()
# model_2_augmentation()
# model_3_pretrained()
model_4_pretrained_augmentation()

print('소요시간 : {:.2f}초'.format(time.time() - start))

# load_history(version=1)
# load_history(version=2)
# load_history(version=3)
# load_history(version=4)

# load_model(version=1)
# load_model(version=2)
# load_model(version=3)     # error
