# Day_33_02_tfhub.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub

# mobilenet-v2
def get_image_classifier():
    url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4'

    # mobilenet = tf.keras.models.load_model('mobilenet_v2')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[224, 224, 3]))
    model.add(hub.KerasLayer(url))
    # model.add(mobilenet)

    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )

    # labels = np.array(open(labels_path).readlines())
    # print(labels)     #
    # labels = [w.split() for w in labels]

    # labels = np.array(open(labels_path).read().splitlines())
    labels = np.array(open(labels_path).read().split())
    print(labels)       # ['background' 'tench' 'goldfish' ... 'ear' 'toilet' 'tissue']


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

array_scaled = array_hopper / 255
# array_scaled = array_hopper / 127 # bright
# array_scaled = array_hopper / 510   # dark

model, labels = get_image_classifier()

preds = model.predict(array_hopper)
print(preds)

# plt.subplot(1, 2, 2)
# plt.title('scaled')
# plt.imshow(array_scaled)
# plt.show()


