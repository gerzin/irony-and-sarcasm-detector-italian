"""
This module contains utility functions to process a Dataframe or Serie
"""
from functools import reduce


def extract_symbols(col):
    """
    Extracts all the different symbols in a column of the dataframe.
    :param col: column of the dataframe from which extract the symbols.
    :return: A set containing all the symbols.
    """
    tmp = col.map(set)
    symbols = reduce(lambda a, b: a.union(b), tmp)
    return symbols
