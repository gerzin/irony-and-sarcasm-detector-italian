"""
This module defines the dataset preprocessing pipeline.
"""
import pandas as pd
import functools
import preprocessing.tweetutils as tweetutils


def remove_urls(df, col_name='text'):
    """
    Removes the URLs from a column.

    :param df: dataframe to clean.
    :param col_name: column to remove urls from.
    :return: The dataframe without URLs in col_name.
    """
    df[col_name] = df[col_name].apply(tweetutils.remove_urls)
    return df


def extract_hashtags(df, col_name='text'):
    """
    Augments the dataframe with a column containing a list of hashtags present in the col_name column.

    :param df: dataframe to augment.
    :param col_name: column to look up hashtags from.
    :return: The augmented dataframe containing a column for the hashtags.
    """
    df['hashtags'] = df[col_name].apply(tweetutils.find_hashtags)
    return df


def lowercase(df, col_name='text'):
    df[col_name] = df[col_name].str.lower()
    return df


def replace_emoji(df, col_name='text'):
    def handle_emoji(s):
        new_str = tweetutils.substitute_emoji(s)
        return new_str.replace("_", " ")

    df[col_name] = df[col_name].apply(handle_emoji)
    return df


def delete_numbers(df, col_name='text'):
    df[col_name] = df[col_name].apply(tweetutils.remove_numbers)
    return df


def remove_hashtags(df, col_name='text'):
    """
    Removes the hashtag symbol from a column.

    :param df: dataframe to remove '#' from.
    :param col_name: column to remove '#' from.
    :return: a dataframe without hashtags in the column col_name.
    """
    df[col_name] = df[col_name].apply(lambda x: x.replace("#", ""))
    return df


def handle_mentions(df, col_name='text'):
    """
    Handles the mentions in the tweets by replacing them with a token.

    :param df: dataframe containing the tweets.
    :param col_name: name of the column to remove the mentions from.
    :return: a dataframe with the mentions replaced with a token in the column col_name.
    """
    df[col_name] = df[col_name].apply(tweetutils.remove_mentions)
    return df


class ItalianTweetsPreprocessingPipeline:

    def __init__(self, transformations=None):
        """
        Pipeline to preprocess the dataset.

        :param transformations: list of functions contained in the pipeline.
                                If None the default transformations applied are:
                                remove_urls, handle_mentions, extract_hashtags, remove_hashtags,
                                replace_emoji, delete_numbers, lowercase
        """
        if transformations is None:
            transformations = [
                remove_urls,
                handle_mentions,
                extract_hashtags,
                remove_hashtags,
                replace_emoji,
                delete_numbers,
                lowercase
            ]
        self.transformations = transformations

    def apply(self, df):
        """
        Run a dataframe through the pipeline's steps.
        :param df: dataframe to process.
        :return: the transformed dataframe.
        """
        return functools.reduce(lambda _df, trans: _df.pipe(trans), self.transformations, df)

    def add_transformation(self, transf):
        """
        Appends a transformation to the list of transformations.
        :param transf: function that takes a dataframe as first argument, processes it and returns a dataframe.
        :return: None
        """
        self.transformations.append(transf)