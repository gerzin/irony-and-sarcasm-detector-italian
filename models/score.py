from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Activation, ActivityRegularization, Embedding, Dense, Bidirectional, Dropout, Flatten, LSTM, \
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, GRU
from keras.callbacks import EarlyStopping
from keras.activations import relu
from sklearn.utils import class_weight
import numpy as np
import keras.backend as K
from keras_self_attention import SeqSelfAttention
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import tensorflow as tf


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_score(y_true, y_pred):
    pairs = list(zip(y_true, y_pred))
    tp = pairs.count((1, 1))
    fp = pairs.count((0, 1))
    fn = pairs.count((1, 0))

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def get_embedding_layer(experiment, word_index, embedding_matrix=None):
    if "embedding_file" in experiment:
        return Embedding(
            len(word_index) + 1,
            experiment["embedding_dimension"],
            weights=[embedding_matrix],
            trainable=experiment["embedding_trainable"],
            input_length=experiment["max_length"])
    else:
        return Embedding(
            len(word_index) + 1,
            experiment["embedding_dimension"],
            input_shape=(experiment["max_length"],))


def create_model(experiment, X_train, y_train, embedding_matrix=None, word_index=None):
    model = Sequential()

    if experiment["model"] in ["lstm", "cnn", "nn"]:
        model.add(get_embedding_layer(experiment, word_index, embedding_matrix=embedding_matrix))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        if 'attention' in experiment and experiment['attention']:
            model.add(SeqSelfAttention(
                attention_width=15,
                attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                attention_activation=None,
                kernel_regularizer=regularizers.l2(1e-6),
                use_attention_bias=False,
                attention_regularizer_weight=1e-4,
                name='Attention',
            ))
    if experiment["model"] == "mlp":
        model.add(Dense(5000, input_shape=(X_train.shape[1],)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2500))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    elif experiment["model"] == "lstm":
        model.add(Bidirectional(GRU(8)))
        model.add(Dropout(0.5))
    elif experiment["model"] == "cnn":
        model.add(Conv1D(64, 8, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Flatten())
        model.add(Dropout(0.5))
    elif experiment["model"] == "nn":
        model.add(Flatten())
        model.add(Dense(1000))
        model.add(Dropout(0.5))
        model.add(Dense(500))
        model.add(Dropout(0.5))
    if 'regularization' in experiment and experiment['regularization']:
        model.add(ActivityRegularization(l1=0.001, l2=0.0001))
    model.add(Dense(y_train.shape[1], activation="softmax"))

    # model.summary()
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train):
    callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
    y_ints = [y.argmax() for y in y_train]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_ints),
                                                      y_ints)
    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=10,
                        shuffle=True,
                        verbose=0,
                        # callbacks=callbacks,
                        # validation_split=0.1,
                        class_weight=class_weights
                        )
    return model
