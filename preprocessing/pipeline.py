from sklearn.pipeline import Pipeline
import re


def remove_urls(s: str, sub='', strip=True):
    """
    Removes urls from strings.
    :param s: String to clean.
    :param sub: What to substitute the urls with (default = '').
    :param strip: If True the returned string is stripped. (default = True)
    :return: The string obtained by replacing the urls in the original string with the value of the sub parameter.
    """
    cleaned = re.sub(r"http\S+", sub, s)
    return cleaned.strip() if strip else cleaned


class ItalianTweetsPreprocessingPipeline:

    def __init__(self, *args, **kwargs):
        self.pipeline = Pipeline(

        )
