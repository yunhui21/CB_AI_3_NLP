# Day_34_03_exam_3.py
import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 자격증 문제 3
# 학습 데이터로 학습하고 검사 데이터로 정확도를 구하세요. (90%)


# rps: rock, paper, scissor
# url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
# urllib.request.urlretrieve(url, 'rps/rps.zip')
#
# zip_ref = zipfile.ZipFile('rps/rps.zip', 'r')
# zip_ref.extractall('rps')
# zip_ref.close()
#
# url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip'
# urllib.request.urlretrieve(url, 'rps/rps-test-set.zip')
#
# zip_ref = zipfile.ZipFile('rps/rps-test-set.zip', 'r')
# zip_ref.extractall('rps')
# zip_ref.close()

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
    'rps/rps',
    batch_size=batch_size,
    target_size=(150, 150),
    class_mode='sparse'
)
valid_flow = valid_gen.flow_from_directory(
    'rps/rps-test-set',
    batch_size=batch_size,
    target_size=(150, 150),
    class_mode='sparse'
)

conv_base = tf.keras.applications.VGG16(
    include_top=False,
    # input_shape=[150, 150, 3]
    )

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[150, 150, 3]))
model.add(conv_base) # 통과하기전에 이미지 증식을 할수있다.

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit_generator(
    train_flow,
    epochs=1,
    steps_per_epoch= train_flow.samples // train_flow.batch_size,
    validation_data=valid_flow,
    # validation_steps=batch_size,
    verbose=2
)

