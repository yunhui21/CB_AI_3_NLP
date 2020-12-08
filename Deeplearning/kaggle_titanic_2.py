# kaggle_titanic_2.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd


# 문제 1
# 지금까지 배운 기술을 모두 적용해서 최적화된 모델을 만드세요
# (결측치/앙상블 제외)

# 문제 2
# 멀티 레이어 코드를 함수로 분리하세요

# 문제 3
# train 데이터를 train과 validation으로 나눠서 모델을 최적화 한 다음에
# test 데이터에 적용한 결과를 업로드하세요

# 문제 4
# 성별(Sex) 컬럼을 추가하세요
# LabelEncoder와 LabelBinarizer 중에서 좋은 것을 사용하세요

def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv', index_col=0)
    print(titanic, end='\n\n')

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    le = preprocessing.LabelEncoder()
    Sex = le.fit_transform(titanic.Sex)
    # print(Sex.shape)
    # print(Sex[:5])

    # lb = preprocessing.LabelBinarizer()
    # Sex = lb.fit_transform(titanic.Sex)
    # print(Sex.shape)
    # print(Sex[:5])

    Sex = np.eye(2, dtype=np.float32)[Sex]
    # print(Sex.shape)
    # print(Sex[:5])

    titanic.drop(['Age', 'Cabin', 'Embarked'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    titanic.info()

    x = np.hstack([titanic.values[:, 1:], Sex])
    y = titanic.values[:, :1]

    return x, np.float32(y)


def get_data_test():
    titanic = pd.read_csv('kaggle/titanic_test.csv', index_col=0)

    le = preprocessing.LabelEncoder()
    Sex = le.fit_transform(titanic.Sex)
    Sex = np.eye(2, dtype=np.float32)[Sex]

    titanic.drop(['Age', 'Cabin', 'Embarked'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    x = np.hstack([titanic.values, Sex])
    ids = titanic.index.values

    return x, ids


def make_submission(file_path, ids, preds):
    f = open(file_path, 'w', encoding='utf-8')

    print('PassengerId,Survived', file=f)

    for i in range(len(ids)):
        result = int(preds[i] > 0.5)
        print('{},{}'.format(ids[i], result), file=f)

    f.close


def multi_layers_1(ph_x, ph_d):
    w1 = tf.get_variable(str(np.random.rand()), shape=[x_train.shape[1], 7],
                         initializer=tf.glorot_uniform_initializer)
    b1 = tf.Variable(tf.zeros([7]))

    w2 = tf.get_variable(str(np.random.rand()), shape=[7, 4],
                         initializer=tf.glorot_uniform_initializer)
    b2 = tf.Variable(tf.zeros([4]))

    w3 = tf.get_variable(str(np.random.rand()), shape=[4, 1],
                         initializer=tf.glorot_uniform_initializer)
    b3 = tf.Variable(tf.zeros([1]))

    z1 = tf.matmul(ph_x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_prob=ph_d)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_prob=ph_d)

    z3 = tf.matmul(d2, w3) + b3

    return z3, tf.sigmoid(z3)


def multi_layers_2(ph_x, ph_d, input_size, layers):
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


def model_titanic(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)
    ph_d = tf.placeholder(tf.float32)       # drop-out

    # z, hx = multi_layers_1(ph_x, ph_d)
    # z, hx = multi_layers_2(ph_x, ph_d, input_size=x_train.shape[1], layers=[7, 4, 1])
    z, hx = multi_layers_2(ph_x, ph_d, input_size=x_train.shape[1], layers=[15, 11, 7, 4, 1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 1000
    batch_size = 32
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

        if i % 10 == 0:
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

# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(x)

x = scaler.transform(x)
x_test = scaler.transform(x_test)

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.7)

# print(x_train.shape, y_train.shape)     # (623, 4) (623, 1)
# print(x_valid.shape, y_valid.shape)     # (268, 4) (268, 1)

preds_valid, preds_test = model_titanic(x_train, y_train, x_valid, x_test)

show_accuracy(preds_valid, y_valid.reshape(-1))
make_submission('kaggle/titanic_submission.csv', ids, preds_test)
