# kaggle_titanic_3.py
import numpy as np
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pandas as pd

# 문제 1
# 결측치 컬럼인 Age 컬럼에 포함된 결측치를 평균으로 채워주세요
# 그리고 결과가 어떻게 나오는지 확인하세요

# 문제 2
# Fare와 Embarked 필드의 결측치를 채우세요 (최빈값)

# 문제 3
# 앙상블을 적용하세요

def get_data_train():
    titanic = pd.read_csv('kaggle/titanic_train.csv', index_col=0)
    # print(titanic, end='\n\n')

    # print(titanic['Age'].mean())
    # print(titanic['Age'].median())
    # print(titanic['Age'].mode())

    # print(titanic['Embarked'].mode())

    # print(titanic['Embarked'].unique())         # ['S' 'C' 'Q' nan]
    # print(titanic['Embarked'].value_counts())   # S 644, C 168, Q 77
    #
    # print(titanic['Fare'].unique())
    # print(titanic['Fare'].value_counts())

    # titanic.fillna(0, inplace=True)

    # train에는 Embarked 결측치, test에는 Fare 결측치
    titanic.Age.fillna(28, inplace=True)        # mean
    titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(8.05, inplace=True)     # mode

    # titanic.info()
    # exit(-1)

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    le = preprocessing.LabelEncoder()
    Sex = le.fit_transform(titanic.Sex)
    # print(Sex.shape)
    # print(Sex[:5])

    lb = preprocessing.LabelBinarizer()
    Embarked = lb.fit_transform(titanic.Embarked)
    # Sex = lb.fit_transform(titanic.Sex)
    # print(Sex.shape)
    # print(Sex[:5])

    Sex = np.eye(2, dtype=np.float32)[Sex]
    # print(Sex.shape)
    # print(Sex[:5])

    titanic.drop(['Cabin', 'Embarked'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    titanic.info()

    x = np.hstack([titanic.values[:, 1:], Sex, Embarked])
    y = titanic.values[:, :1]

    return x, np.float32(y)


def get_data_test():
    titanic = pd.read_csv('kaggle/titanic_test.csv', index_col=0)

    # train에는 Embarked 결측치, test에는 Fare 결측치
    titanic.Age.fillna(28, inplace=True)        # mean
    titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(8.05, inplace=True)     # mode

    le = preprocessing.LabelEncoder()
    Sex = le.fit_transform(titanic.Sex)
    Sex = np.eye(2, dtype=np.float32)[Sex]

    lb = preprocessing.LabelBinarizer()
    Embarked = lb.fit_transform(titanic.Embarked)

    titanic.drop(['Cabin', 'Embarked'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    x = np.hstack([titanic.values, Sex, Embarked])
    ids = titanic.index.values

    return x, ids


# 기존 컬럼을 활용해서 새로운 컬럼 생성
def get_data_train_added():
    titanic = pd.read_csv('kaggle/titanic_train.csv', index_col=0)
    # print(titanic, end='\n\n')

    # print(titanic['Age'].mean())
    # print(titanic['Age'].median())
    # print(titanic['Age'].mode())

    # print(titanic['Embarked'].mode())

    # print(titanic['Embarked'].unique())         # ['S' 'C' 'Q' nan]
    # print(titanic['Embarked'].value_counts())   # S 644, C 168, Q 77
    #
    # print(titanic['Fare'].unique())
    # print(titanic['Fare'].value_counts())

    # print(titanic['Pclass'].value_counts())     # 3 491, 1 216, 2 184

    # titanic.fillna(0, inplace=True)

    # train에는 Embarked 결측치, test에는 Fare 결측치
    titanic.Age.fillna(28, inplace=True)        # mean
    titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(8.05, inplace=True)     # mode

    titanic['Family'] = titanic.SibSp + titanic.Parch + 1
    titanic['IsAlone'] = 1
    titanic['IsAlone'].loc[titanic.Family > 1] = 0

    titanic['AgeBin'] = pd.cut(titanic.Age, 5, labels=[0, 1, 2, 3, 4])
    titanic['FareBin'] = pd.qcut(titanic.Fare, 4, labels=[0, 1, 2, 3])

    # 직접 구간을 설정할 때
    # titanic.loc[titanic.Age < 15, 'Age'] = 0
    # titanic.loc[(titanic.Age >= 15) & (titanic.Age < 25), 'Age'] = 1

    # titanic.info()
    # exit(-1)

    # print(titanic.Sex.unique())
    # print(titanic.Ticket.unique())
    #
    # print(titanic.Sex.value_counts())
    # print(titanic.Ticket.value_counts())

    le = preprocessing.LabelEncoder()
    Sex = le.fit_transform(titanic.Sex)
    # print(Sex.shape)
    # print(Sex[:5])

    lb = preprocessing.LabelBinarizer()
    Embarked = lb.fit_transform(titanic.Embarked)
    Pclass = lb.fit_transform(titanic.Pclass)
    # Sex = lb.fit_transform(titanic.Sex)
    # print(Sex.shape)
    # print(Sex[:5])

    Sex = np.eye(2, dtype=np.float32)[Sex]
    # print(Sex.shape)
    # print(Sex[:5])

    titanic.drop(['Cabin', 'Embarked', 'Pclass'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    titanic.info()

    x = np.hstack([titanic.values[:, 1:], Sex, Embarked, Pclass])
    y = titanic.values[:, :1]

    return x, np.float32(y)


def get_data_test_added():
    titanic = pd.read_csv('kaggle/titanic_test.csv', index_col=0)

    # train에는 Embarked 결측치, test에는 Fare 결측치
    titanic.Age.fillna(28, inplace=True)        # mean
    titanic.Embarked.fillna('S', inplace=True)  # mode
    titanic.Fare.fillna(8.05, inplace=True)     # mode

    titanic['Family'] = titanic.SibSp + titanic.Parch + 1
    titanic['IsAlone'] = 1
    titanic['IsAlone'].loc[titanic.Family > 1] = 0

    titanic['AgeBin'] = pd.cut(titanic.Age, 5, labels=[0, 1, 2, 3, 4])
    titanic['FareBin'] = pd.qcut(titanic.Fare, 4, labels=[0, 1, 2, 3])

    le = preprocessing.LabelEncoder()
    Sex = le.fit_transform(titanic.Sex)
    Sex = np.eye(2, dtype=np.float32)[Sex]

    lb = preprocessing.LabelBinarizer()
    Embarked = lb.fit_transform(titanic.Embarked)
    Pclass = lb.fit_transform(titanic.Pclass)

    titanic.drop(['Cabin', 'Embarked', 'Pclass'], axis=1, inplace=True)
    titanic.drop(['Name', 'Sex', 'Ticket'], axis=1, inplace=True)

    x = np.hstack([titanic.values, Sex, Embarked, Pclass])
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
    z, hx = multi_layers_2(ph_x, ph_d, input_size=x_train.shape[1], layers=[15, 9, 9, 1])

    loss_i = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    # optimizer = tf.train.AdamOptimizer(0.001)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100
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
        #     print(i, c / n_iteration)

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


# x, y = get_data_train()
# x_test, ids = get_data_test()

x, y = get_data_train_added()
x_test, ids = get_data_test_added()

# scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
scaler.fit(x)

x = scaler.transform(x)
x_test = scaler.transform(x_test)

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.8)

# print(x_train.shape, y_train.shape)     # (623, 4) (623, 1)
# print(x_valid.shape, y_valid.shape)     # (268, 4) (268, 1)

results_1 = np.zeros([len(x_test)])
n_ensembles = 7
for i in range(n_ensembles):
    preds_valid, preds_test = model_titanic(x_train, y_train, x_valid, x_test)

    show_accuracy(preds_valid, y_valid.reshape(-1))
    results_1 += preds_test

results_2 = np.zeros([len(x_test)])
for i in range(n_ensembles):
    _, preds_test = model_titanic(x, y, x_test, x_test)
    results_2 += preds_test

print('-' * 30)
results_1 /= n_ensembles
results_2 /= n_ensembles

make_submission('kaggle/titanic_submission_1.csv', ids, results_1)
make_submission('kaggle/titanic_submission_2.csv', ids, results_2)

# 수정
# train size ; 0.8
# epochs : 200
# batch size : 16
# optimizer : rms prop
# fare bin : qcut
# multi layers : 15, 9, 9, 1
