# Day_28_01_chosun.py
import tensorflow as tf
import numpy as np
import re
from sklearn import preprocessing

# url = 'https://raw.githubusercontent.com/greentec/greentec.github.io/master/public/other/data/chosundynasty/corpus.txt'
# file_path = tf.keras.utils.get_file(
#     'chosun.txt', url, cache_dir='.', cache_subdir='data')
# print(file_path)

def clean_str(string):

    string = re.sub(r"[^가-힣0-9]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()


def get_data():
    f = open('data/chosun.txt', 'r', encoding= 'utf-8')
    long_text = f.read()
    long_text = clean_str(long_text)
    long_text = long_text[:1000]
    f.close()

    tokens = long_text.split()
    vocab  = sorted(set(tokens)) + ['UNK'] # paddin을 UNK로 대체

    return tokens, vocab


    # 문제
    # 조선왕조실록을 학습하는 모델을 구축하세요.


def model_chosun_1():
    # 파일읽기 -> 불용어 제거 -> 숫자로 전환 -> x, y로 분리
    tokens, vocab = get_data()
    # print(tokens[:5])           # ['태조', '이성계', '선대의', '가계', '목조']
    # tokens을 숫자로 숫자를 token으로 바꾸는

    word2idx = {w:i for i, w in enumerate(vocab)}
    # idx2word = {i:w for i, w in enumerate(vocab)}
    idx2word = np.array(vocab)
    # print(list(word2idx.items())[:5])
    # [('001', 0), ('002', 1), ('003', 2), ('004', 3), ('10대', 4)]
    # print(idx2word[:5])
    # ['001' '002' '003' '004' '10대']

    tokens_idx = [word2idx[w] for w in tokens] # tokens의 인덱스를 가지고
    # print(tokens_idx[:5]) # [194, 149, 104, 10, 74]

    seq_len, vocab_size =  25, len(vocab)
    # print(len(vocab))
    # exit(-1)
    x = [tokens_idx[i:i+seq_len] for i in range(len(tokens)-seq_len)]
    y = [tokens_idx[i+seq_len] for i in range(len(tokens)-seq_len)]
    # print(x[0])     # [194, 149, 104, 10, 74, 154, 174, 99, 140, 18,...
    # print(y[:10])   # [47, 155, 166, 46, 30, 145, 34, 207, 46, 149]

    # 19_03_kerasnavermovie.py
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[seq_len]))
    model.add(tf.keras.layers.Embedding(vocab_size, 100))      # 입력(2차원), 출력(3차원)
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    model.summary()
    # exit(-1)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=10, batch_size=128, verbose=2)
    model.save('data/chosun_model_1.h5')

    sent = '동헌에 나가 활을 쐈다'
    act_like_writer_1(sent, model, word2idx, idx2word, seq_len)
    # act_like_writer_2(sent, model, word2idx, idx2word, seq_len)


def model_chosun_2():
    tokens, vocab = get_data()

    word2idx = {w:i for i, w in enumerate(vocab)}
    idx2word = np.array(vocab)

    tokens_idx = [word2idx[w] for w in tokens] # tokens의 인덱스를 가지고

    seq_len, vocab_size =  25, len(vocab)

    # x = [tokens_idx[i:i+seq_len] for i in range(len(tokens)-seq_len)]
    # y = [tokens_idx[i+seq_len] for i in range(len(tokens)-seq_len)]

    # tokens_idx를 사용할 크기로 잘라내야 한다.
    # Datasets 생성-> 사용 크기로 분할 (seq_len + 1 (26개씩 자른다.))->(25, 1)로 재구성  -> shuffle
    # -> 배치크기로 분할 (128)
    sent_slices = tf.data.Dataset.from_tensor_slices(tokens_idx) # 객체생성
    # print(type(sent_slices))        # <class 'tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter'>
    # print(sent_slices.take(2)) # tensorflow시험에 나오는 배점이 큰 영역
    #                                 # <DatasetV1Adapter shapes: (), types: tf.int32>
    #
    # for takes in sent_slices.take(2):
    #     print(takes.numpy(), takes)

    # 이전 코드는 슬라이싱으로 밀기 때무에 모든 토큰에 대해 예측하지만
    # 25개의 토큰으로 1개를 예측하고, 다음 번 25로 넘어가기 때문에 예측 횟수에 차이가 많이 발생한다.
    # (최종 데이터 갯수가 엄청나게 줄어든다.)

    sent_sequences = sent_slices.batch(seq_len+1, drop_remainder=True)  # 자투리처리 -> 버린다.
    # for takes in sent_sequences.take(2):
    #     print(takes.numpy())
    # [194 149 104  10  74 154 174  99 ...]
    # [155 166  46  30 145  34 207  46 ...]
    # for takes in sent_sequences.take(2):
    #     print(takes.numpy())

    sent_xy = sent_sequences.map(lambda  chunk: (chunk[:-1], chunk[-1]))
    # for xx, yy in sent_xy.take(2):
    #     print(xx.numpy(), yy.numpy())
    # [194 149 104  10  74 154 174  99 ...]
    # [155 166  46  30 145  34 207  46 ...]

    steps_per_epoch = len(tokens_idx) // (seq_len + 1)
    sent_shuffled = sent_xy.shuffle(buffer_size=steps_per_epoch)
    # for xx, yy in sent_shuffled.take(2):
    #     print(xx.numpy(), yy.numpy())
    # [155 166  46  30 145  34 207...]
    # [117  91  98  48 207 163 136...]
    # tensorflow 자격시험 5번 문제에 나옴

    # 문제
    # 앞쪽 2개만 출력하세요.
    sent_batches = sent_shuffled.batch(128)
    # for xx, yy in sent_batches.take(2):
    #     print(xx.shape, yy.shape)
    # 데이터 갯수가 작아서 1개밖에 없다.
    # (12, 25) (12,)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[seq_len]))
    model.add(tf.keras.layers.Embedding(vocab_size, 100))      # 입력(2차원), 출력(3차원)
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['acc'])

    model.fit(sent_batches.repeat(), epochs=100,
              steps_per_epoch=steps_per_epoch, verbose=2)
    model.save('data/chosun_model_2.h5')

    sent = '동헌에 나가 활을 쐈다'
    act_like_writer_1(sent, model, word2idx, idx2word, seq_len)
    # act_like_writer_2(sent, model, word2idx, idx2word, seq_len)
    # 27_01 파일 참고
    # layer = model.get_layer(0)
    # print(layer.weights)


def act_like_writer_1(sent, model, word2idx, idx2word, seq_len):
    current = sent.split()

    # if len(current) < seq_len:
    #     current = current + ['UNK'] * (seq_len - len(current))
    # 25개로 고정을 해보자. 25개가 넘어가면 제외한다.

    for i in range(100):
        tokens = current[-seq_len:] # 마지막 시퀀스 25개.. 25개가 안되면 전체를 갖고 온다.
        tokens_idx = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in tokens]
        # 누락되는 단어는 unk롤 대체하겠다. 존재한느것만 바꾼다. 25개 아니다.
        # 숫자로 만들어준다.

        tokens_pad = tf.keras.preprocessing.sequence.pad_sequences(
            [tokens_idx], maxlen=seq_len, value= word2idx['UNK']    # tokens_idx 1차원 [tokens_idx]이차원
        )
        # 25개로 다시 맞춰준다.
        # print(tokens_pad)

        preds = model.predict(tokens_pad)
        # print(preds.shape)
        preds_arg = np.argmax(preds[0])
        # print(preds_arg)

        current.append(idx2word[preds_arg]) # pred_arg는 바로더해줄수없다 tokens이라서
        # 가장 큰 값을

    print(current)
    # ['동헌에', '나가', '활을', '쐈다', '백성', '듣고', '생겼다', '양무', .....


def act_like_writer_2(sent, model, word2idx, idx2word, seq_len):
    current = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in sent.split()]

    filled = seq_len - len(current)
    if filled > 0:
        current = [word2idx['UNK']] * filled + current
    else:
        filled = 0
    # 25개로 고정을 해보자. 25개가 넘어가면 제외한다.
    # print(current)
    for i in range(100):
        tokens_idx = current[-seq_len:]
        tokens_pad = np.int32([tokens_idx])
        preds = model.predict_classes(tokens_pad)
        # print(preds)

        preds_arg = preds[0]        # 32
        print(i, preds_arg, idx2word[preds_arg])
        current.append(preds_arg)

    print(idx2word[current[filled:]])


def load_model(sent, model_path):
    tokens, vocab = get_data()

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = np.array(vocab)

    model = tf.keras.models.load_model(model_path)

    act_like_writer_2(sent, model, word2idx, idx2word, 25)


# model_chosun_1()
model_chosun_2()
# load_model('이성계의 아들 이방원이 왕이 되다.', 'data/chosun_model_1.h5')