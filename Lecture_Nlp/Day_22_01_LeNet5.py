# Day_22_01_LeNet5.py
import tensorflow as tf

# 문제
# LeNet5 모델을 구현해서, 최초 만들었던 cnn 모델과 정확도를 비교하세요

# 문제
# summary 함수에서 출력한 Total params 항목이 61,706으로 되어 있는데
# 왜 그렇게 되는지 직접 계산해 보세요

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28, 1]))
model.add(tf.keras.layers.Conv2D(6, [5, 5], [1, 1], 'same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))
model.add(tf.keras.layers.Conv2D(16, [5, 5], [1, 1], 'valid', activation='relu'))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)
print('acc :', model.evaluate(x_test, y_test, verbose=0))
