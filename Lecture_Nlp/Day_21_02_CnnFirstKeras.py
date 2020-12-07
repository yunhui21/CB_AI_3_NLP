# Day_21_02_CnnFirstKeras.py
import tensorflow as tf
import numpy as np

# 문제
# Day_21_01_CnnFirst.py 파일의 내용을 케라스 버전으로 수정하세요


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ------------------------------------- #

# w1 = tf.Variable(tf.random.normal([3, 3, 1, 32]))
# b1 = tf.Variable(tf.zeros([32]))
#
# w2 = tf.Variable(tf.random.normal([3, 3, 32, 64]))
# b2 = tf.Variable(tf.zeros([64]))
#
# w3 = tf.Variable(tf.random.normal([7 * 7 * 64, 128]))
# b3 = tf.Variable(tf.zeros([128]))
#
# w4 = tf.Variable(tf.random.normal([128, 10]))
# b4 = tf.Variable(tf.zeros([10]))
#
# # ------------------------------------- #
#
# ph_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# ph_y = tf.placeholder(tf.int32)
#
# c1 = tf.nn.conv2d(ph_x, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
# r1 = tf.nn.relu(c1 + b1)
# p1 = tf.nn.max_pool2d(r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# c2 = tf.nn.conv2d(p1, filter=w2, strides=[1, 1, 1, 1], padding='SAME')
# r2 = tf.nn.relu(c2 + b2)
# p2 = tf.nn.max_pool2d(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# print(c1.shape, p1.shape)       # (60000, 28, 28, 32) (60000, 14, 14, 32)
# print(c2.shape, p2.shape)       # (60000, 14, 14, 64) (60000, 7, 7, 64)
#
# flat = tf.reshape(p2, shape=[-1, p2.shape[1] * p2.shape[2] * p2.shape[3]])
# print(flat.shape)               # (60000, 3136)
#
# # (60000, 128) = (60000, 3136) @ (3136, 128)
# d3 = tf.matmul(flat, w3) + b3
# r3 = tf.nn.relu(d3)
#
# # (60000, 10) = (60000, 128) @ (128, 10)
# z = tf.matmul(r3, w4) + b4
# hx = tf.nn.softmax(z)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28, 1]))
# model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3],
#                                  strides=[1, 1], padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same'))
model.add(tf.keras.layers.Conv2D(32, [3, 3], [1, 1], 'same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))
model.add(tf.keras.layers.Conv2D(64, [3, 3], [1, 1], 'same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=ph_y)
# loss = tf.reduce_mean(loss_i)
#
# optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
# train = optimizer.minimize(loss)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
#
# epochs = 10
# batch_size = 100
# n_iteration = len(x_train) // batch_size
#
# for i in range(epochs):
#     total = 0
#     for j in range(n_iteration):
#         n1 = j * batch_size  # 0 100 200 300
#         n2 = n1 + batch_size  # 100 200 300 400
#
#         xx = x_train[n1:n2]
#         yy = y_train[n1:n2]
#
#         sess.run(train, {ph_x: xx, ph_y: yy})
#         total += sess.run(loss, {ph_x: xx, ph_y: yy})
#
#     print(i, total / n_iteration)

model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)

# preds = sess.run(hx, {ph_x: x_test})
preds = model.predict(x_test)
preds_arg = np.argmax(preds, axis=1)

print('acc :', np.mean(preds_arg == y_test))  # acc : 0.9727

