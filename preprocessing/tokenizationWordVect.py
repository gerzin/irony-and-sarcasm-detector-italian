import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
from config import Config
import sqlite3




def tokenize_frame(tweets, mode = "sostituisciTag"):
    """
    Tokenize the tweets and return a frame of sequences.

    :param tweets: column of the dataframe containing the tweets
    :return: a DataFrame of sequences ready to be fed to a neural network
    """
    
    
    PADDING_LEN = 65
    
    maxLen = 0
    sequences = []
    for t in tweets:
        seq = word_tokenize(t)
        to_del = []
        for j in range(len(seq)):
            if seq[j] == '<':
                if j+2 < len(seq):
                    if seq[j+2] == '>':
                        temp = seq[j]
                        temp = temp  + seq[j+1]
                        temp = temp + seq[j+2]
                        seq[j] = temp
                        to_del.append(j+1)
                        to_del.append(j+2)

        if to_del != []:
            to_del.sort(reverse = True)
            for e in to_del:
                del seq[e]
            
        sequences.append(seq)
        if len(seq) > maxLen:
            maxLen = len(seq)
            
       
        

    

    set_stopwords = stopwords.words(Config.TEXT_LANGUAGE)

    sentences_clear = []
    for sentence in sequences:
        sentence_clear = []
        for word in sentence:
            if word not in set_stopwords and word not in string.punctuation:
                sentence_clear.append(word)
        sentences_clear.append(sentence_clear)
 

    conn = sqlite3.connect(str(Config.WORD_EMBEDDING_PATH))
    c = conn.cursor()
    
    sentences_vectorized = []
    if mode == "sostituisciTag":
        for s in sentences_clear:

            sentence_vectorized = []
            for w in s:
                c.execute(f'SELECT * FROM STORE WHERE KEY = "{w}"')
                r = (c.fetchall())
                if r != []:
                    sentence_vectorized.append(list((r[0])[1:-1]))
                else:
                    if w == "<num>":
                        c.execute('SELECT * FROM STORE WHERE KEY = "numero"')
                        r = (c.fetchall())
                        sentence_vectorized.append(list((r[0])[1:-1]))
                    elif w == "<men>":
                        c.execute('SELECT * FROM STORE WHERE KEY = "menzione"')
                        r = (c.fetchall())
                        sentence_vectorized.append(list((r[0])[1:-1]))   
                    else:
                        sentence_vectorized.append([0]*128)
            sentences_vectorized.append(sentence_vectorized)
    elif mode == "ignoraTag":
        for s in sentences_clear:

            sentence_vectorized = []
            for w in s:
                c.execute(f'SELECT * FROM STORE WHERE KEY = "{w}"')
                r = (c.fetchall())
                if r != []:
                    sentence_vectorized.append(list((r[0])[1:-1]))
                else:
                    sentence_vectorized.append([0]*128)
            sentences_vectorized.append(sentence_vectorized)


    for s in sentences_vectorized:
        if len(s) < PADDING_LEN:
            for i in range(PADDING_LEN - len(s)):
                s.append([0]*128)


    return sentences_vectorized
