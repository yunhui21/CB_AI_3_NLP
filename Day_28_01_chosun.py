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

    x = [tokens_idx[i:i+seq_len] for i in range(len(tokens)-seq_len)]
    y = [tokens_idx[i+seq_len] for i in range(len(tokens)-seq_len)]
    # print(x[0])     # [194, 149, 104, 10, 74, 154, 174, 99, 140, 18,...
    # print(y[:10])   # [47, 155, 166, 46, 30, 145, 34, 207, 46, 149]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[seq_len]),
        tf.keras.layers.Embedding(vocab_size, 100),  # 입력(2차원), 출력(3차원)
        tf.keras.layers.LSTM(128, return_sequences=False),
        tf.keras.layers.Dense(vocab_size, activation='softmax'),
    ])
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['acc'])

    model.fit(x, y, epochs=10, batch_size=128, verbose=2)
    model.save('data/chosun_model_1.h5')

    # sent = '동헌에 나가 활을 쐈다'
    # act_like_writer_1(sent, model, word2idx, idx2word, seq_len)
    # act_like_writer_2(sent, model, word2idx, idx2word, seq_len)


def act_like_writer_1(sent, model, word2idx, idx2word, seq_len):
    current = sent.split()

    # if len(current) < seq_len:
    #     current = current + ['UNK'] * (seq_len - len(current))
    # 25개로 고정을 해보자. 25개가 넘어가면 제외한다.

    for i in range(100):
        tokens = current[-seq_len:]
        tokens_idx = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in tokens]
        # 누락되는 단어는 unk롤 대체하겠다. 존재한느것만 바꾼다. 25개 아니다.
        # 숫자로 만들어준다.

        tokens_pad = tf.keras.preprocessing.sequence.pad_sequences(
            [tokens_idx], maxlen=seq_len, value= word2idx['UNK']
        )
        # 25개로 다시 맞춰준다.
        print(tokens_pad)

        preds = model.predict(tokens_pad)
        print(preds.shape)
        preds_arg = np.argmax(preds[0])
        print(preds_arg)

        current.append(idx2word[preds_arg]) # pred_arg는 바로더해줄수없다 tokens이라서
        # 가장 큰 값을

    print(current)


def act_like_writer_2(sent, model, word2idx, idx2word, seq_len):
    current = [word2idx[w] if w in word2idx else word2idx['UNK'] for w in sent.split()]

    if len(current) < seq_len:
        current = current + [word2idx['UNK']] * (seq_len - len(current))
    # 25개로 고정을 해보자. 25개가 넘어가면 제외한다.
    # print(current)
    for i in range(100):
        tokens_idx = current[-seq_len:]

        preds = model.predict_classes([tokens_idx])
        print(preds)

        preds_arg = preds[0]        # 32
        # print(preds)              # [32]

        current.append(preds_arg)

    print(current)


model_chosun_1()
