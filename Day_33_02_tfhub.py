# Day_33_02_tfhub.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# mobilenet-v2
def get_image_classifier():
    url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[224, 224, 3]))
    model.add(hub.KerasLayer(url))

    # mobilenet = tf.keras.models.load_model('mobilenet_v2')
    # model.add(mobilenet)

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )

    # labels = np.array(open(labels_path).readlines())
    # print(labels)     #
    # labels = [w.split() for w in labels]

    labels = np.array(open(labels_path).read().splitlines())
    # labels = np.array(open(labels_path).read().split())
    print(labels)       # ['background' 'tench' 'goldfish' ... 'ear' 'toilet' 'tissue']

    return model, labels

def classify_image():
    # img_url = 'https://storage.googleapis.com/download.tensoflow.org/example_images/grace_hopper.jpg'
    img_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg'
    img_path = tf.keras.utils.get_file('grace_hopper.jpg', img_url)
    print(img_path)

    # 문제
    # 다운로드된 이미지를 출력하세요.
    img_hopper = Image.open(img_path).resize([224, 224])
    # plt.imshow(img_hopper)
    # plt.show()

    array_hopper = np.array(img_hopper)
    print(array_hopper.shape)           # (224, 224, 3)

    plt.subplot(1, 2, 1)
    plt.title('original')
    plt.imshow(array_hopper)

    # print(np.min(array_hopper), np.max(array_hopper))   # 0 255

    array_scaled = array_hopper / 255       # minmax scaling
    # array_scaled = array_hopper / 127 # bright
    # array_scaled = array_hopper / 510   # dark

    model, labels = get_image_classifier()

    # preds = model.predict(array_hopper) # 차원이 아니어서 에러
    # preds = model.predict([array_hopper]) #
    # preds = model.predict(array_hopper[np.newaxis]) # array_hopper는 오리지널.
    preds = model.predict(array_scaled[np.newaxis])

    print(preds)
    # ['background' 'tench' 'goldfish' ... 'bolete' 'ear' 'toilet tissue']

    preds_arg = np.argmax(preds[0])
    print(preds_arg, labels[preds_arg])
    # [[ 0.2231094   0.05557813  0.3937732  ... -0.90115947 -1.3781339 3.4246128 ]] 722 pillow


    plt.subplot(1, 2, 2)
    plt.title('scaled : {}'.format(labels[preds_arg]))
    plt.imshow(array_scaled)
    plt.show()


def classify_by_generator():

    # img_url = 'https://storage.googleapis.com/download.tensoflow.org/example_images/grace_hopper.jpg'
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
    # for xx, yy in data_flow:
    #     print(xx.shape, yy.shape)       # (32, 224, 224, 3) (32,)

    xx, yy = data_flow.next()   # Found 3670 images belonging to 5 classes.
    # print(xx.shape, yy.shape)   # (32, 224, 224, 3) (32,)
    # print(yy)                   # [0. 3. 1. 3. 3. 1. 3. 3. 2. 1. 3. 4. 3. 1...] 5개

    model, labels = get_image_classifier()
    preds = model.predict(xx)   # ['background' 'tench' 'goldfish' ... 'bolete' 'ear' 'toilet tissue']
    preds_arg = np.argmax(preds, axis=1)
    # print(preds_arg)            # [986 884 986 986 503 947 579 717 947 986 986 329 986 ...]
    # print(labels[preds_arg])    # ['coral fungus' 'daisy' 'swab' 'Bedlington terrier' 'lemon' 'daisy'...]


    plt.figure(figsize=(12, 6))
    for i, (img, label, pred) in enumerate(zip(xx, yy, preds_arg)):

        plt.subplot(4, 8, i+1)
        plt.title(labels[pred])
        plt.axis('off')
        plt.imshow(img)

    plt.tight_layout
    plt.show()

# classify_image()
classify_by_generator()