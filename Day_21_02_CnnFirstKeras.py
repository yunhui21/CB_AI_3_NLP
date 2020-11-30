# Day_21_02_CnnFirstKeras.py
# Day_21_01_CnnFirst.py
import tensorflow as tf
import numpy as np

# 문제
# Day_21_01_CnnFirst.py 파일의 내용을 케라스 버전으로 수정하세요.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)      # (60000,) (10000,)

print(x_train.dtype, y_train.dtype)     # uint8 uint8
print(np.min(x_train), np.max(x_train)) #0 255

# x_train = np.float32(x_train)
# x_test  = np.float32(x_test)
x_train = x_train/255
x_test  = x_test/255

x_train = x_train.reshape(-1, 28, 28, 1) # # (60000, 28, 28)
x_test  = x_test.reshape(-1, 28, 28, 1)

#---------------------------------------#
#
# w1 = tf.Variable(tf.random.normal([3, 3, 1, 32]))
# b1 = tf.Variable(tf.zeros([32]))
#
# w2 = tf.Variable(tf.random.normal([3, 3, 1, 64]))
# b2 = tf.Variable(tf.zeros([64]))
#
# w3 = tf.Variable(tf.random.normal([7 * 7 * 64, 128]))
# b3 = tf.Variable(tf.zeros([128]))
#
# w4 = tf.Variable(tf.random.normal([128, 10]))
# b4 = tf.Variable(tf.zeros([10]))
#
#
# #---------------------------------------#
#
# ph_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# ph_y = tf.placeholder(tf.int32)
#
# c1 = tf.nn.conv2d(x_train, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
# r1 = tf.nn.relu(c1 + b1)
# p1 = tf.nn.max_pool2d(r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# c2 = tf.nn.conv2d(p1, filter=w2, strides=[1, 1, 1, 1], padding='SAME')
# r2 = tf.nn.relu(c2 + b2)
# p2 = tf.nn.max_pool2d(r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# print(c1.shape, p1.shape)       # (60000, 28, 28, 32) (60000, 14, 14, 32)
# print(c2.shape, p2.shape)       # (60000, 14, 14, 32) (60000, 7, 7, 32)
#
# flat = tf.reshape(p2, shape=[-1, p2.shape[1] * p2.shape[2] * p2.shape[3]])
# print(flat.shape)               # (60000, 3136)
#
# # (60000< 128) = (60000, 3136) @ (3136, 128)
# d3 = tf.matmul(flat, w3) + b3
# r3 = tf.nn.relu(d3)
#
# z = tf.matmul(r3, w4) + b4
# hx = tf.nn.softmax(z)

#---------------------------------------------#
# Day_13_01_RnnMnist.py

model = tf.keras.Sequential()
#
model.add(tf.keras.layers.Input(shape=[28,28,1]))
model.add(tf.keras.layers.Conv2D(32, [3, 3],[1, 1], 'same', activation='relu')
model.add(tf.keras.layers.MaxPooling2D([2, 2], [2, 2], 'same')
model.add(tf.keras.layers.Conv2D(64, [3,3], [1,1], 'same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D([2, 2], [2, 2], 'same')

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))






# loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=ph_y)
# loss = tf.reduce_mean(loss_i)
#
# optimizer = tf.train.AdamOptimizer(0.001)
# train = optimizer.minimize(loss)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])
# sess = tf.compat.v1.S,itializer())
#
# epochs = 10
# batch_size = 100 # 한번에 처리할 데이터개수
# n_iteration = len(x_train) // batch_size # epochs한번에 배치사이즈만큼의 횟수
#
#
# for i in range(epochs):
#     total = 0
#     for j in range(n_iteration): # 60000의 데이터 100으로 나누어
#         n1 = j * batch_size         # 0, 100, 200, 300
#         n2 = n1 + batch_size        # 100, 200, 300, 400
#
#         xx = x_train[n1:n2]
#         yy = y_train[n1:n2]
#
#         sess.run(train, {ph_x: xx, ph_y: yy})
#         total += sess.run(loss,  {ph_x:xx, ph_y:yy})
#
#     print(i, total/n_iteration ) # batch
model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2)

# preds = sess.run(z, {ph_x: x_test})
# print(preds.shape)

preds = model.predict(x_test)
preds_arg = np.argmax(preds, axis=1)
print('acc:', np.mean(preds_arg == y_test))
