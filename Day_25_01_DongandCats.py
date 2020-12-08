# Day_25_01_DongandCats.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_model_name(version):
    filename = 'dogcat_small_{}.h5'.format(version)
    return os.path.join('dogs_and_cats', filename)

def get_history_name(version):
    filename = 'dogcat_small_{}.history'.format(version)
    return os.path.join('dogs_and_cats', filename)

def show_history(history, version):
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], 'r', label='train')
    plt.plot(history['val_loss'], 'g', label='valid')
    plt.title('loss {}'.format(version))
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(history['acc'], 'r', label='train')
    plt.plot(history['val_acc'], 'g', label='valid')
    plt.title('acc {}'.format(version))
    plt.legend()

    plt.show()

def save_history(history, version):
    with open(get_history_name(version), 'wb') as f:
        pickle.dump(history.history, f)

def load_history(version):
    with open(get_history_name(version), 'rb') as f:
        history = pickle.load(f)
        show_history(history.history, version)

def load_model(version):
    model = tf.keras.models.load_model(get_model_name(version))
    # model.summary()

    # 문제
    # 나머지 코드를 구현하세요.
    test_gen = ImageDataGenerator(rescale=1/255)

    test_flow = test_gen.flow_from_directory(
        'Dogsandcats/small/test',
        batch_size=1000,
        target_size=(150, 150),
        class_mode='binary'
    )

    # print('acc:', model.evaluate_generator(test_flow, steps=1, verbose=0))

    x_test, y_test = test_flow()
    print('acc:', model.evaluate(x_test, y_test, verbose=0))



def model_1_baseline():
    data_gen = ImageDataGenerator(rescale=1/255)

    batch_size = 32
    train_flow = data_gen.flow_from_directory(
        'DogsandCats/small/train',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    valid_flow = data_gen.flow_from_directory(
        'DogsandCats/small/validation',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    # test_flow = data_gen.flow_from_directory(
    #     'DogsandCats/small/test',
    #     batch_size=(150, 150),
    #     class_mode='binary'
    # )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[150, 150, 3]))

    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit_generator(
        train_flow,
        epochs=1,
        steps_per_epoch=2000 // batch_size,
        validation_data=valid_flow,
        # validation_steps=batch_size,
        verbose=2
    )

    model.save(get_model_name(version=1))
    save_history(history, version=1)


def model_2_augmentation():
    train_gen = ImageDataGenerator(
        rescale=1/255,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=-.1,
        zoom_range=0.1,
        rotation_range=20,

    )
    valid_gen = ImageDataGenerator(
        rescale=1 / 255,
    )

    batch_size = 32
    train_flow = train_gen.flow_from_directory(
        'dogs_and_cats_model/small/train',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    valid_flow = valid_gen.flow_from_directory(
        'dogs_and_cats_model/small/validation',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    # test_flow = data_gen.flow_from_directory(
    #     'DogsandCats/small/test',
    #     batch_size=(150, 150),
    #     class_mode='binary'
    # )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[150, 150, 3]))

    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))
    model.add(tf.keras.layers.Conv2D(128, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(2, 2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit_generator(
        train_flow,
        epochs=100,
        steps_per_epoch=2000 // batch_size,
        validation_data=valid_flow,
        # validation_steps=batch_size,
        verbose=2
    )

    model.save(get_model_name(version=2))
    save_history(history, version=2)


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

            if n2 > sample_count:
                needed = sample_count - n1 # 삐져나간만큼
                x[n1:] = conv_base.predict(xx[:needed])
                y[n1:] = yy[:needed]
                break



            x[n1:n2] = conv_base.predict(xx) # 피쳐를 만드는 핵심코드, predict를 통과한 값이 어떤 의미를 갖는지 알아야 한다.
            y[n1:n2] = yy
        return x.reshape(-1, 4 * 4 * 512), y #

    def extract_features_2(conv_base, data_gen, directory, sample_count, batch_size):

        flow = data_gen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary'
        )
        x, y = [], []
        n_loops = sample_count // batch_size # 남는 자투리는 다시 처리를 해본다.
        for ii, (xx, yy) in enumerate(flow):

            if ii >= n_loops:
                # print(n_loops * batch_size)
                needed = sample_count - n_loops * batch_size
                x.append(conv_base.predict(xx[:needed]))
                y.append(yy[:needed])
                break

            x.append(conv_base.predict(xx)) # 피쳐를 만드는 핵심코드, predict를 통과한 값이 어떤 의미를 갖는지 알아야 한다.
            y.append(yy)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, newshape = (-1,))
        # print(x.shape, y.shape)

        return x.reshape(-1, 4 * 4 * 512), y #

    conv_base = tf.keras.applications.VGG16(
        include_top=False,
        input_shape=[150, 150, 3])
    conv_base.summary()

    # return
    batch_size = 32
    data_gen = ImageDataGenerator(rescale=1/255)

    # x_train, y_train = extract_features(conv_base, data_gen, 'dogs_and_cats/small/train', 2000, batch_size)
    # x_valid, y_valid = extract_features(conv_base, data_gen, 'dogs_and_cats/small/validation', 1000, batch_size)
    # x_test, y_test   = extract_features(conv_base, data_gen, 'dogs_and_cats/small/test', 1000, batch_size)

    x_train, y_train = extract_features_2(conv_base, data_gen, 'dogs_and_cats/small/train', 2000, batch_size)
    x_valid, y_valid = extract_features_2(conv_base, data_gen, 'dogs_and_cats/small/validation', 1000, batch_size)
    x_test, y_test   = extract_features_2(conv_base, data_gen, 'dogs_and_cats/small/test', 1000, batch_size)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[4*4*512]))

    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])
    history = model.fit(
        x_train, y_train,
        epochs = 10,
        batch_size=batch_size,
        verbose=2,
        validation_data=(x_valid, y_valid)
    )

    # 모델을 저장하기는 하지만 다른 모델과 호환은 되지 않는다.(입력 모앙 다름)
    model.save(get_model_name(version=3))
    save_history(history, 3)

    print('acc:', model.evaluate(x_test, y_test))


def model_4_pretrained_augmentation():
    train_gen = ImageDataGenerator(
        rescale=1 / 255,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=-.1,
        zoom_range=0.1,
        rotation_range=20,

    )
    valid_gen = ImageDataGenerator(
        rescale=1 / 255,
    )

    batch_size = 32
    train_flow = train_gen.flow_from_directory(
        'dogs_and_cats_model/small/train',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    valid_flow = valid_gen.flow_from_directory(
        'dogs_and_cats_model/small/validation',
        batch_size=batch_size,
        target_size=(150, 150),
        class_mode='binary'
    )
    # test_flow = data_gen.flow_from_directory(
    #     'DogsandCats/small/test',
    #     batch_size=(150, 150),
    #     class_mode='binary'
    # )
    conv_base = tf.keras.applications.VGG16(
        include_top=False,
        # input_shape=[150, 150, 3]
        )
    conv_base.trainable = False

    # for layer in conv_base.layers:
    #     print(layer.name)
    #
    #     if 'bloc5' in layer.name:
    #         layer.tainable = False

    # weight update 기본
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[150, 150, 3]))

    model.add(conv_base) # 통과하기전에 이미지 증식을 할수있다.

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    history = model.fit_generator(
        train_flow,
        epochs=100,
        steps_per_epoch=2000 // batch_size,
        validation_data=valid_flow,
        # validation_steps=batch_size,
        verbose=2
    )

    model.save(get_model_name(version=4))
    save_history(history, 4)


# model_1_baseline()
# model_2_augmentation()
# model_3_pretrained()
# model_4_pretrained_augmentation()


# load_history(version=1)
# load_history(version=2)
# load_history(version=3)
# load_history(version=4)

# load_model(version=1)
# load_model(version=2)
# load_model(version=3)
# load_model(version=4)