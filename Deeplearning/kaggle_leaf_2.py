# kaggle_leaf_2.py
import numpy as np
from sklearn import model_selection, preprocessing, decomposition
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2                  # opencv-python

# 문제 1
# 타이타닉 2번의 코드를 가져다가 나뭇잎에 맞게 수정하세요
# (서브미션 파일 생성은 제외)

# 문제 2
# 서브미션 파일을 생성하는 함수를 만드세요

# 문제 3
# 나뭇잎 이미지 폴더에 있는 파일 이름을 읽어와서 파일 경로를 구성하세요

# 문제 4
# 앞에서 읽어온 경로를 사용해서 이미지에 대한 피처를 만드세요
# 피처 : 너비, 높이, 면적, 비율, 수직/수평

# 문제 5
# 새롭게 만든 기본 피처를 기존 피처에 연결하세요

# 문제 6
# 새롭게 만든 pca 피처를 기존 피처에 연결하세요

# 문제 7
# 새롭게 만든 moments 피처를 기존 피처에 연결하세요


def make_basic_features():
    dir_path = 'kaggle/leaf_images/'

    features = {}
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        # file_path = dir_path + '/' + filename
        # print(file_path)        # kaggle/leaf_images/46.jpg

        leaf_id = int(filename.split('.')[0])
        # print(leaf_id)

        img = plt.imread(file_path)
        # print(filename)
        # print(type(img), img.shape)     # <class 'numpy.ndarray'> (379, 689)

        h, w = img.shape
        area = h * w
        ratio = h / w
        orientation = int(h > w)

        features[leaf_id] = (h, w, area, ratio, orientation)

    return features


# 1. 데이터사이언스 pca
# 2. pca에 전달되는 데이터는 반드시 2차원
#    이미지 1장을 1차원으로 변환해서 모든 이미지를 저장하면 2차원이 됩니다
# 3. 차원 갯수는 35로 합니다
# 4. 원본 이미지는 (50, 50)으로 크기 변경 후에 사용합니다
def make_pca_features():
    dir_path = 'kaggle/leaf_images/'

    ids, images = [], []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)

        leaf_id = int(filename.split('.')[0])

        img = Image.open(file_path)
        img = img.resize([50, 50])
        img = np.uint8(img).reshape(-1)

        ids.append(leaf_id)
        images.append(img)

    # print(np.array(images).shape)       # (1584, 2500)

    pca = decomposition.PCA(n_components=35)
    pca_features = pca.fit_transform(images)

    # print(pca_features.shape)           # (1584, 35)

    features = {}
    for leaf_id, feat in zip(ids, pca_features):
        features[leaf_id] = feat

    return features


def make_moments_features():
    dir_path = 'kaggle/leaf_images/'

    features = {}
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        leaf_id = int(filename.split('.')[0])

        img = cv2.imread(file_path, 0)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        moments = cv2.moments(cnt)
        # print(moments)

        features[leaf_id] = tuple(moments.values())

    return features


def get_data_train():
    leaf = pd.read_csv('kaggle/leaf_train.csv', index_col=0)
    # print(leaf, end='\n\n')

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(leaf.species)

    leaf.drop(['species'], axis=1, inplace=True)
    x = leaf.values

    return x, y, le.classes_, leaf.index.values


def get_data_test():
    leaf = pd.read_csv('kaggle/leaf_test.csv', index_col=0)
    return leaf.values, leaf.index.values


def make_submission(file_path, ids, preds, leaf_names):
    f = open(file_path, 'w', encoding='utf-8')

    print('id', *leaf_names, sep=',', file=f)

    for leaf_id, pred in zip(ids, preds):
        print(leaf_id, *pred, sep=',', file=f)

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

    return z, tf.nn.softmax(z)


def model_leaf(x_train, y_train, x_valid, x_test):
    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.int32)
    ph_d = tf.placeholder(tf.float32)       # drop-out

    z, hx = multi_layers(ph_x, ph_d, input_size=x_train.shape[1], layers=[160, 128, 99])

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_y, logits=z)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = 100
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
    return preds_valid, preds_test


def show_accuracy_sparse(preds, labels):
    preds_arg = np.argmax(preds, axis=1)

    equals = (preds_arg == labels)
    print('acc :', np.mean(equals))


def append_new_features(ids, origin, new_features):
    # print(type(origin), type(new_features))     # <class 'numpy.ndarray'> <class 'dict'>

    # 퀴즈
    # 아래 코드를 컴프리헨션으로 바꾸세요
    added = [new_features[leaf_id] for leaf_id in ids]

    # added = []
    # for leaf_id in ids:
    #     added.append(new_features[leaf_id])

    # print(np.array(added).shape)        # (990, 5)

    return np.hstack([origin, added])


x, y, leaf_names, ids_train = get_data_train()
x_test, ids_test = get_data_test()

features_basic = make_basic_features()
features_pca = make_pca_features()
features_moments = make_moments_features()

# print(x.shape, x_test.shape)            # (990, 192) (594, 192)

x = append_new_features(ids_train, x, features_basic)
x_test = append_new_features(ids_test, x_test, features_basic)

x = append_new_features(ids_train, x, features_pca)
x_test = append_new_features(ids_test, x_test, features_pca)

x = append_new_features(ids_train, x, features_moments)
x_test = append_new_features(ids_test, x_test, features_moments)

print(x.shape, x_test.shape)
# basic : (990, 197) (594, 197)
# pca   : (990, 232) (594, 232)

scaler = preprocessing.StandardScaler()
scaler.fit(x)

x = scaler.transform(x)
x_test = scaler.transform(x_test)

x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, train_size=0.7)

# print(x_train.shape, y_train.shape)     # (693, 192) (693,)
# print(x_valid.shape, y_valid.shape)     # (297, 192) (297,)

preds_valid, preds_test = model_leaf(x_train, y_train, x_valid, x_test)

show_accuracy_sparse(preds_valid, y_valid)
make_submission('kaggle/leaf_submission_1.csv', ids_test, preds_test, leaf_names)

# preds_max = np.max(preds_test, axis=1)
# sorted_max = np.sort(preds_max)
#
# print(sorted_max[:10])
# print(sorted_max[-10:])

eye = np.eye(99, dtype=np.float32)
for i in range(len(preds_test)):
    p = preds_test[i]
    m = np.max(p)

    if m > 0.98:
        n = np.argmax(p)
        preds_test[i] = eye[n]

make_submission('kaggle/leaf_submission_2.csv', ids_test, preds_test, leaf_names)


