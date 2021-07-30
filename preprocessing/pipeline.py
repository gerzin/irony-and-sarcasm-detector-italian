import pandas as pd
import datautils


def remove_urls(df, col_name='text'):
    """
    Removes the URLs from a column.

    :param df: dataframe to clean.
    :param col_name: column to remove urls from.
    :return: The dataframe without URLs in col_name.
    """
    df[col_name] = df[col_name].apply(datautils.remove_urls)
    return df


def extract_hashtags(df, col_name='text'):
    """
    Augments the dataframe with a column containing a list of hashtags present in the col_name column.

    :param df: dataframe to augment.
    :param col_name: column to look up hashtags from.
    :return: The augmented dataframe containing a column for the hashtags.
    """
    df['hashtags'] = df[col_name].apply(datautils.find_hashtags)
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


class ItalianTweetsPreprocessingPipeline:

    def __init__(self, transformations=None):
        """
        Pipeline to preprocess the dataset.

        :param transformations: list of functions contained in the pipeline.
                                If None the default transformations applied are:
                                remove_urls, extract_hashtags
        """
        if transformations is None:
            transformations = [
                remove_urls,
                extract_hashtags,
                remove_hashtags
            ]
        self.transformations = transformations

    def apply(self, df: pd.DataFrame):
        """
        Run a dataframe through the pipeline's steps.
        :param df: dataframe to process.
        :return: the transformed dataframe.
        """
        return (df.pipe(self.transformations[0])
                .pipe(self.transformations[1])
                .pipe(self.transformations[2])
                )
