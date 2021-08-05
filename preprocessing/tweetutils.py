"""
This module contains utility functions to process a single tweet in the form of a string.
"""
import re
import emoji


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


def remove_digits(s: str, number_tok="<NUM>"):
    """

    :param number_tok:
    :param s:
    :return:
    """
    return re.sub(r'[0-9]+', number_tok + " ", s)


def substitute_emoji(s: str):
    """

    :param s:
    :return:
    """
    return emoji.demojize(s, delimiters=(" ", " "), language='it')


def remove_mentions(s: str, mention_tok="<MEN>"):
    """

    :param mention_tok:
    :param mention_sub:
    :param s:
    :return:
    """
    s = re.sub(r"(\[@*\w+\])", mention_tok, s)
    return re.sub(r"(@\w+)", mention_tok, s)
