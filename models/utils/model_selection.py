from sklearn.model_selection import train_test_split


def train_test_validation_split(df, train_split=0.8, seed=42):
    """
    Splits the tweets dataset into train, test and validation sets.

    Assumes that we want an equal-sized validation and test set.

    :param df: Dataframe with features in column 'text' and labels in columns 'irony', 'sarcasm'.
    :param train_split: size of the train set.
    :param seed: seed for the rng.
    :return: (X_train, y_train), (X_test, y_test), (X_val, y_val)
    """

    features = df['text']
    targets = df[['irony', 'sarcasm']]

    X_train, X, y_train, y = train_test_split(features, targets, train_size=train_split, random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(X, y, train_size=0.5, random_state=seed)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)
