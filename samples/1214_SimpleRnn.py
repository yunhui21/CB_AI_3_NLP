from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt


max_features = 1000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train.shape)            #(25000, )

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=20)
x_test  = preprocessing.sequence.pad_sequences(x_test, maxlen=20)
print(x_train.shape)            # (25000, 20)

model = tf.keras.Sequential()


def model_1():
    model.add(tf.keras.layers.Embedding(max_features, 8, input_length = maxlen))
    model.add(tf.keras.layers.SimpleRNN(10))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    model.summary()


def model_SimpleRNN():
    model.add(tf.keras.layers.Embedding(1000, 32))
    model.add(tf.keras.layers.SimpleRNN(24, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(40))
    model.summary()


def model_LSTM():
    model.add(tf.keras.layers.Embedding(1000, 32))
    model.add(tf.keras.layers.LSTM(24, activation='relu'))
    model.summary()

def model_LSTM_layer():
    model.add(tf.keras.layers.Embedding(max_features, 32, input_length=maxlen))
    model.add(tf.keras.layers.LSTM(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

def model_GRU():

    model.add(tf.keras.layers.Embedding(1000, 32))
    model.add(tf.keras.layers.GRU(24, activation='relu'))
    model.summary()


model.compile(
    optimizer=tf.keras.optimizers.RMSprop,
    loss = tf.keras.losses.binary_crossentropy,
    metrics = ['accuracy']
)
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)


# model_1()
# model_SimpleRNN()
# model_LSTM()
model_LSTM_layer()
# model_GRU()