# Day_22_02_MultiLayers.py
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)
    # y_arg = np.argmax(labels, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


mnist = input_data.read_data_sets('mnist')
# mnist = input_data.read_data_sets('mnist', one_hot=True)

print(mnist.train.images.shape)         # (55000, 784)
print(mnist.validation.images.shape)    # (5000, 784)
print(mnist.test.images.shape)          # (10000, 784)

print(mnist.train.labels.shape)         # (55000,)

print(mnist.train.images.dtype)         # float32
print(mnist.train.labels.dtype)         # uint8

x_train = mnist.train.images
y_train = np.int32(mnist.train.labels)
x_test = mnist.test.images
y_test = np.int32(mnist.test.labels)

# 문제
# train 데이터로 학습하고 test 데이터로 정확도를 계산하세요
n_features = x_train.shape[1]
n_classes = 10
w = tf.Variable(tf.random_uniform([n_features, n_classes]))     # (784, 10)
b = tf.Variable(tf.random_uniform([n_classes]))                 # (10,)

ph_x = tf.placeholder(tf.float32)
ph_y = tf.placeholder(tf.int32)

# (55000, 10) = (55000, 784) @ (784, 10)
z = tf.matmul(ph_x, w) + b
hx = tf.nn.softmax(z)

loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_y, logits=z)
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 10
batch_size = 100
n_iteration = len(x_train) // batch_size    # 550

for i in range(epochs):
    c = 0
    for j in range(n_iteration):
        n1 = j * batch_size
        n2 = n1 + batch_size

        xx = x_train[n1:n2]
        yy = y_train[n1:n2]

        sess.run(train, {ph_x: xx, ph_y: yy})
        c += sess.run(loss, {ph_x: xx, ph_y: yy})

    print(i, c / n_iteration)

preds_train = sess.run(hx, {ph_x: x_train})
preds_test = sess.run(hx, {ph_x: x_test})

show_accuracy_sparse(preds_train, y_train)
show_accuracy_sparse(preds_test, y_test)

sess.close()



