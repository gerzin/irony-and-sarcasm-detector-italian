import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize_frame(sentences):
    '''Tokenizer'''
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    '''Rimozione delle stopwords'''
    set_stopwords = stopwords.words('italian')

    sentences_clear = []
    for sentence in sentences:
        sentence_clear = []
        for word in sentence:
            if word not in set_stopwords and word not in string.punctuation:
                sentence_clear.append(word)
        sentences_clear.append(sentence_clear)

    '''Trasformazione di ogni parola in un intero'''
    # word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences_clear)

    '''Padding di ogni frase alla stessa dimensione'''

    padded = pad_sequences(sequences, padding='post')
    return padded
