import tensorflow as tf
from transformers import TFBertModel, AutoTokenizer

EMBEDDING_SIZE = 128


def get_bert_gru_classifier(hidden_layers, compile_model=True):
    """
    Returns the BERT GRU model.
    params:
    hidden_layers - list containing specification in form of a tuple (number of neurons, return sequence, dropout)
                    for each GRU hidden layer to add.
    compile_model - flag indicating if to compile_model the model or not.

    example:
        model = get_bert_gru_classifier([(l1_neurons, l1_retseq, l1_dropout),...,(ln_neurons, ln_retseq, ln_dropout)])
    """
    model_url = "dbmdz/bert-base-italian-xxl-cased"
    bert = TFBertModel.from_pretrained(model_url)
    input_ids_in = tf.keras.layers.Input(shape=(128,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(128,), name='masked_token', dtype='int32')

    embedding_layer = bert(input_ids_in, attention_mask=input_masks_in)[0]

    first = True
    for layer in hidden_layers:
        if first:
            X = tf.keras.layers.GRU(layer[0], return_sequences=layer[1])(embedding_layer)
            first = False
        else:
            X = tf.keras.layers.GRU(layer[0], return_sequences=layer[1])(X)
        if layer[2] != 0.0:
            X = tf.keras.layers.Dropout(layer[2])(X)

    X = tf.keras.layers.Dense(2, activation='sigmoid')(X)

    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs=X)

    for layer in model.layers[:3]:
        layer.trainable = False
    if compile_model:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model
