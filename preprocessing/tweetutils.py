"""
This module contains utility functions to process a single tweet in the form of a string.
"""
import re
import emoji

NUMBER_TOKEN = "<NUM>"
MENTION_TOKEN = "<MEN>"


def remove_urls(s: str, sub='', strip=True):
    """
    Removes urls from a string.

    :param s: String to clean.
    :param sub: What to substitute the urls with (default = '').
    :param strip: If True the returned string is stripped. (default = True)
    :return: The string obtained by replacing the urls in the original string with the value of the sub parameter.
    """
    cleaned = re.sub(r"http\S+", sub, s)
    return cleaned.strip() if strip else cleaned


def find_hashtags(s: str):
    """
    Finds the hashtags in a string.

    :param s: String from which extract the hashtags.
    :return: list of the hashtags contained in s (without the #).
    """
    return re.findall(r"#(\w+)", s)


def remove_numbers(s: str, number_tok=NUMBER_TOKEN):
    """
    Removes the numbers from a string replacing them with a token.

    :param s: string to clean.
    :param number_tok: token to replace the numbers with.
    :return:
    """
    return re.sub(r'[0-9]+', number_tok + " ", s)


def substitute_emoji(s: str):
    """
    Substitutes the emojis in a string with a string indicating their meaning.

    :param s: string to clean.
    :return: a string in which the emojis are replaced with a string.
    """
    return emoji.demojize(s, delimiters=(" ", " "), language='it')


def remove_mentions(s: str, mention_tok=MENTION_TOKEN):
    """
    Removes the mentions from a tweet and replaces them with a token.

    :param s: string to clean.
    :param mention_tok: token to replace the mentions with
    :return: string in which the mentions have been replaced by a token.
    """
    s = re.sub(r"(\[@*\w+\])", mention_tok, s)
    return re.sub(r"(@\w+)", mention_tok, s)
