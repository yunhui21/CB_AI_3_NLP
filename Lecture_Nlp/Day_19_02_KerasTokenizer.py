# Day_19_02_KerasTokenizer.py
import tensorflow as tf

long_text = ("if you want to build a ship,"
             " don't drum up people to collect wood and"
             " don't assign them tasks and work,"
             " but rather teach them to long"
             " for the endless immensity of the sea.")

# 10개를 전달하면 패딩을 제외한 9개만 사용
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10)

# tokenizer.fit_on_texts(long_text)
# print(tokenizer.index_word)     # {1: 't', 2: 'o', 3: 'e', ...}
# print(tokenizer.word_index)     # {'t': 1, 'o': 2, 'e': 3, ...}

tokenizer.fit_on_texts(long_text.split())
print(tokenizer.index_word)     # {1: 'to', 2: "don't", 3: 'and', ...}
print(tokenizer.word_index)     # {'to': 1, "don't": 2, 'and': 3, ...}
print()

# print(tokenizer.index_word[0])    # 에러. 0번 인덱스는 패딩으로 사용
print(tokenizer.oov_token)          # None. oov(out of vocabulary)
tokenizer.oov_token = '*'
print(tokenizer.oov_token)          # *
print()

print(tokenizer.index_word[9])      # build
print(tokenizer.index_word[10])     # a
print(tokenizer.index_word[11])     # ship
print()

print(tokenizer.texts_to_sequences(['build', 'a', 'ship']))     # [[9], [], []]
print(tokenizer.texts_to_sequences([['build', 'a', 'ship']]))   # [[9]]
print()

# 문제
# sequences_to_texts 함수에 전달할 배열을 만드세요
sequences = [[3, 9, 2, 4, 7], [1, 4, 13, 8], [10, 15]]
print(tokenizer.sequences_to_texts(sequences))
# ["and build don't them you", 'to them want', '']
print()

print(tf.keras.preprocessing.sequence.pad_sequences(sequences))
print()

print(tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='pre'))
print(tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post'))
print()

print(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=4))
print(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=4, truncating='pre'))
print(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=4, truncating='post'))

