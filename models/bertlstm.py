from modelsconfig import ModelsConfig
from transformers import TFBertModel, AutoTokenizer
import numpy as np
import tensorflow as tf


def get_bert_tokenizer(model_url=ModelsConfig.BERT_ITA_XXL_CASED, tok_len=ModelsConfig.BERT_TOKENIZER_LENGTH):
    tokenizer = AutoTokenizer.from_pretrained(model_url, add_special_tokens=True, max_length=tok_len,
                                              pad_to_max_length=True)

    return tokenizer


def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                       return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])

    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments,
                                                                                                    dtype='int32')


def get_bert_lstm_classifier(model_url=ModelsConfig.BERT_ITA_XXL_CASED):
    bert = TFBertModel.from_pretrained(model_url)

    input_ids_in = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32')

    embedding_layer = bert(input_ids_in, attention_mask=input_masks_in)[0]

    X = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.Concatenate(axis=-1)([X, embedding_layer])

    X = tf.keras.layers.MaxPooling1D(20)(X)
    X = tf.keras.layers.SpatialDropout1D(0.4)(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(128, activation="relu")(X)
    X = tf.keras.layers.Dropout(0.25)(X)
    X = tf.keras.layers.Dense(2, activation='softmax')(X)

    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)

    for layer in model.layers[:3]:
        layer.trainable = False

    return model
