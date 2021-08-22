import tensorflow as tk
from tensorflow import keras
from config import Config

EMBEDDING_SIZE = 128


def get_nn_gru(num_words: int, embedding_size=EMBEDDING_SIZE, compile_model=True):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(num_words, embedding_size, input_length=Config.SEQUENCE_LENGTH, mask_zero=True))
    model.add(keras.layers.GRU(128, return_sequences=True))
    model.add(keras.layers.GRU(128))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    if compile_model:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
