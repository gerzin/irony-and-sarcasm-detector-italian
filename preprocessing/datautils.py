import re


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
