import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def tokenize_frame(tweets):
    """
    Tokenize the tweets and return a frame of sequences.

    :param tweets: column of the dataframe containing the tweets
    :return: a DataFrame of sequences ready to be fed to a neural network
    """
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(tweets)

    set_stopwords = stopwords.words('italian')

    sentences_clear = []
    for sentence in tweets:
        sentence_clear = []
        for word in sentence:
            if word not in set_stopwords and word not in string.punctuation:
                sentence_clear.append(word)
        sentences_clear.append(sentence_clear)

    sequences = tokenizer.texts_to_sequences(sentences_clear)

    padded = pad_sequences(sequences, padding='post')

    # new_frame = pd.DataFrame()
    # new_frame['Sequences'] = padded

    # return new_frame
    return padded
