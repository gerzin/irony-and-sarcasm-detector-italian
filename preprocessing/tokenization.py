import string
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from config import Config

OOV_TOKEN = "<OOV>"
TOKENIZER_NUM_WORDS = None


class ItalianTweetsTokenizer:
    def __init__(self, preprocessing_pipeline=None, tokenizer=None):
        """
        Tokenizer for the italian tweets dataset.
        :param preprocessing_pipeline: pipeline for preprocessing the dataset, if None the
                                       dataset will not be preprocessed.
        :param tokenizer: fit tokenizer, if None a tokenizer will be fit on the whole dataftame
                          passed to the tokenize function.
        """
        self.preprocessing_pipeline_ = preprocessing_pipeline
        self.tokenizer = tokenizer

    def tokenize(self, df, col_name='text', maxlen=Config.SEQUENCE_LENGTH):
        """
        Tokenizes a dataframe. If a preprocessing pipeline is set, the dataframe is pre-processed first.

        :param df: DataFrame to tokenize
        :param col_name: name of the column containing the strings to tokenize.
        :param maxlen: final length of the sequences
        :return: Tokenized and padded array of fixed-length sequences.
        """
        if self.preprocessing_pipeline_:
            df = self.preprocessing_pipeline_.apply(df)
        if not self.tokenizer:
            tok = Tokenizer(oov_token=OOV_TOKEN, num_words=TOKENIZER_NUM_WORDS)
            tok.fit_on_texts(df[col_name])
            self.tokenizer = tok

        stop_words_set = set(stopwords.words(Config.TEXT_LANGUAGE))
        tweets = df[col_name].tolist()
        pre_tok = [list(filter(lambda x: x not in stop_words_set, tweet.split(" "))) for tweet in tweets]
        sequences = self.tokenizer.texts_to_sequences(pre_tok)
        sequences = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
        return sequences

    def set_preprocessing_pipeline(self, pl):
        """
        Set or replaces the preprocessing pipeline.
        :param pl: pipeline
        :return: None
        """
        self.preprocessing_pipeline_ = pl

    def get_preprocessing_pipeline(self):
        """
        Get the preprocessing pipeline
        :return: the preprocessing pipeline or None if it is not set.
        """
        return self.preprocessing_pipeline_
