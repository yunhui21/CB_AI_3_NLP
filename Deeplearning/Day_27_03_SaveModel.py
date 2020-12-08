# Day_27_03_SaveModel.py
import tensorflow as tf
import pickle


def linear_regression_save():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))

        saver.save(sess, 'model/series', global_step=i)

    # saver = tf.train.Saver()
    # saver.save(sess, 'model/regression')

    print('5 :', sess.run(w * 5 + b))   # 5 : 8.750338
    print('7 :', sess.run(w * 7 + b))   # 7 : 13.501097

    sess.close()


def linear_regression_restore():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(5.0)
    b = tf.Variable(-3.0)

    hx = w * x + b

    loss_i = (hx - y) ** 2
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # for i in range(10):
    #     sess.run(train)
    #     print(i, sess.run(loss))

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    print('5 :', sess.run(w * 5 + b))   # 5 : 8.750338
    print('7 :', sess.run(w * 7 + b))   # 7 : 13.501097

    sess.close()


def pickle_save():
    d = {'age': 23, 'name': 'kim'}

    f = open('model/dict.pkl', 'wb')
    pickle.dump(d, f)
    f.close()


def pickle_restore():
    f = open('model/dict.pkl', 'rb')
    d = pickle.load(f)
    print(d)
    f.close()


# linear_regression_save()
# linear_regression_restore()

pickle_save()
pickle_restore()
