# Day_43_01_tfhub_filne_tune.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 문제
# resnet_50 모델을 사용해서
# 이전 파일에 나온 꽃 데이터에 대해 예측하세요.

#Day_33_02_tfhub.py
def get_image_classifier():
    # classification = top에 있는 dense 레이어 사용
    # tf.keras.applications.VGG16(include_top=True)
    # url = 'https://tfhub.dev/tensorflow/resnet_50/classification/1'

    # feature-vector = top에 있는 dense레이어 사용하지 않고 앞쪽의 컨볼루션만 사용
    # tf.keras.applications.VGG16(include_top=False)
    url = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[224, 224, 3]))
    model.add(hub.KerasLayer(url, trainable=False))             # softmax를 사용함.
    # model.add(tf.keras.layers.Dense(1001, activation='relu'))   # feature_vector 전용
    model.add(tf.keras.layers.Dense(5, activation='softmax'))   # 추가: softmax를 사용해도 gradient vanishing에는 큰 영향을 주지 않으므로 사용ㅎ다.
    model.summary()
    return model

def classify_by_generator():

    img_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    img_path = tf.keras.utils.get_file('flower_photos', img_url, untar=True)
    # print(img_path)

    # 클래스종류가 맞지 않아서 학습을 할수가없다.

    data_gen = ImageDataGenerator(rescale=1 / 255)

    batch_size = 32
    data_flow = data_gen.flow_from_directory(
        img_path,
        batch_size=batch_size,
        target_size=(224, 224),
        class_mode='sparse'
    )

    model = get_image_classifier()

    model.compile(optimizer = tf.keras.optimizers.Adam(),
                   loss = tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=['acc']
    )

    steps_per_epoch = data_flow.samples // batch_size
    model.fit(data_flow, epochs= 2, steps_per_epoch=steps_per_epoch)

    # ---------------------------------------------------------------#
    print(data_flow.class_indices)      # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    labels = sorted([k for k in data_flow.class_indices]) # 전제조건은 정렬되어 있어야 한다.
    labels = np.array(labels)
    print(labels)                       # ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


    xx, yy = data_flow.next()

    preds = model.predict(xx, verbose=0)
    preds_arg = np.argmax(preds, axis=1)
    print(preds_arg)                #[1 3 1 1 3 3 2 0 1 4 3 3 3 3 1 4 3 0 4 4 2 1 4 3 3 0 3 3 1 0 3 0]
    print(labels[preds_arg])        #['dandelion' 'sunflowers' 'dandelion' 'dandelion' 'sunflowers'...]


    plt.figure(figsize=(12, 6))
    for i, (img, label, pred) in enumerate(zip(xx, yy, preds_arg)):
        plt.subplot(4, 8, i+1)
        plt.title(labels[pred])
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout
    plt.show()


# get_image_classifier()
classify_by_generator()
