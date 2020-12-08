# kaggle_CatInTheDat_1.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd

# 문제
# cat in the dat 데이터셋에 대해 캐글에 결과를 등록하세요
# kaggle_titanic_3.py


def get_data_train():
    cat = pd.read_csv('kaggle/cat_train.csv', index_col=0)

    # print(cat)
    # cat.info()

    # total = 0
    # for col in cat.columns:
    #     v = cat[col].unique()
    #     total += len(v)
    #     print(col, len(v), v[:7])
    #
    # print('total :', total)     # total : 16463

    enc = preprocessing.LabelEncoder()

    y = enc.fit_transform(cat.target)
    cat.drop(['target'], axis=1, inplace=True)

    columns = []
    for col in cat.columns:
        columns.append(enc.fit_transform(cat[col]))

    x = np.transpose(columns)

    return x, y.reshape(-1, 1)


def get_data_test():
    cat = pd.read_csv('kaggle/cat_test.csv', index_col=0)

    enc = preprocessing.LabelEncoder()
    columns = [enc.fit_transform(cat[col]) for col in cat.columns]

    return np.transpose(columns), cat.index.values


def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')

    print('id,target', file=f)

    for i, p in zip(ids, preds):
        print('{},{}'.format(i, p), file=f)

    # for i in range(len(ids)):
    #     print('{},{}'.format(ids[i], preds[i]), file=f)

    f.close()


def multi_layers(ph_x, ph_d, input_size, layers):
    n_features = input_size
    d = ph_x
    for n_classes in layers:
        w = tf.get_variable(str(np.random.rand()), shape=[n_features, n_classes],
                            initializer=tf.glorot_uniform_initializer)
        b = tf.Variable(tf.zeros([n_classes]))

        z = tf.matmul(d, w) + b

        if n_classes == layers[-1]:     # 1
            break

        r = tf.nn.relu(z)
        d = tf.nn.dropout(r, keep_prob=ph_d)

        n_features = n_classes

    return z, tf.sigmoid(z)


def model_cat_in_the_dat(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)       # drop-out

    z, hx = multi_layers(ph_x, ph_d, input_size=x_train.shape[1], layers=[1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 10
    batch_size = 16
    n_iteration = len(x_train) // batch_size

    indices = np.arange(len(x_train))

    for i in range(epochs):
        np.random.shuffle(indices)

        c = 0
        for j in range(n_iteration):
            n1 = j * batch_size
            n2 = n1 + batch_size

            part = indices[n1:n2]

            xx = x_train[part]
            yy = y_train[part]

            sess.run(train, {ph_x: xx, ph_y: yy, ph_d: 0.7})
            c += sess.run(loss, {ph_x: xx, ph_y: yy, ph_d: 0.7})

        # if i % 10 == 0:
        print(i, c / n_iteration)

    preds_valid = sess.run(hx, {ph_x: x_valid, ph_d: 1.0})
    preds_test = sess.run(hx, {ph_x: x_test, ph_d: 1.0})

    sess.close()
    return preds_valid.reshape(-1), preds_test.reshape(-1)


# 매개 변수는 모두 1차원.
def show_accuracy(preds, labels):
    preds_1 = (preds > 0.5)
    # y_1 = labels.reshape(-1)

    preds_1 = np.int32(preds_1)
    y_1 = np.int32(labels)

    print('acc :', np.mean(preds_1 == y_1))


x, y = get_data_train()
x_test, ids = get_data_test()

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.8)

print(x_train.shape, y_train.shape)     # (240000, 23) (240000, 1)
print(x_valid.shape, y_valid.shape)     # (60000, 23) (60000, 1)

preds_valid, preds_test = model_cat_in_the_dat(x_train, y_train, x_valid, x_test)
show_accuracy(preds_valid, y_valid.reshape(-1))

make_submission('kaggle/cat_submission.csv', ids, preds_test)
