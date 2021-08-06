import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from config import Config

OOV_TOKEN = "<OOV>"
TOKENIZER_NUM_WORDS = None


def tokenize_frame(tweets):
    """
    Tokenize the tweets and return a frame of sequences.

    :param tweets: column of the dataframe containing the tweets
    :return: a DataFrame of sequences ready to be fed to a neural network
    """
    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(tweets)

    set_stopwords = stopwords.words(Config.TEXT_LANGUAGE)

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


class ItalianTweetsTokenizer:
    def __init__(self, preprocessing_pipeline=None, tokenizer=None):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.tokenizer = tokenizer

    def tokenize(self, df, col_name='text'):
        if self.preprocessing_pipeline:
            df = self.preprocessing_pipeline.apply(df)
        if not self.tokenizer:
            tok = Tokenizer(oov_token=OOV_TOKEN, num_words=TOKENIZER_NUM_WORDS)
            tok.fit_on_texts(df[col_name])
            self.tokenizer = tok

        stop_words_set = set(stopwords.words(Config.TEXT_LANGUAGE))
        tweets = df[col_name].tolist()
        pre_tok = [list(filter(lambda x: x not in stop_words_set, tweet.split(" "))) for tweet in tweets]
        sequences = self.tokenizer.texts_to_sequences(pre_tok)
        sequences = pad_sequences(sequences, padding='post')
        return sequences
