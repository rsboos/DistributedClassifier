import math
import warnings
import numpy as np

from sklearn.model_selection import StratifiedKFold


class P3StratifiedKFold(StratifiedKFold):

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        validation : ndarray
            The validation set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        skf = StratifiedKFold(self.n_splits - 1, self.shuffle, self.random_state)

        for train, test in super(P3StratifiedKFold, self).split(X, y, groups):
            fold_gen = skf.split(X[train, :], y[train], groups)
            _, val1 = next(fold_gen)
            _, val2 = next(fold_gen)

            validation = np.append(val1, val2)
            validation = train[validation]

            train = np.setdiff1d(train, validation)

            yield train, validation, test


class Distributor():
    """Vertically partition data."""

    def __init__(self, n_splits, overlap, random_state=None):
        """Set private propeties.

        Keyword arguments:
            n_splits -- number of splits to be made
            overlap -- percentage of common parts between splits
            random_state -- int or default=None
                If int, random_state is the seed used by the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.
        """
        self.__n_splits = n_splits
        self.__overlap = overlap
        self.__set_random_state(random_state)

    @property
    def n_splits(self):
        return self.__n_splits

    @property
    def overlap(self):
        return self.__overlap

    def split(self, data):
        """Vertically distributes the training data into n parts and returns a list of
        column indexes for each part

        Keyword arguments:
            data -- the training set to be splitted (Data object)
        """
        rs = self.__random_state

        n_features = data.n_features
        n_common = math.ceil(n_features * self.overlap)
        n_distinct = n_features - n_common

        features = range(n_features)
        common_features = rs.choice(n_features, n_common)

        distinct_features = list(set(features) - set(common_features))
        distinct_features = rs.permutation(distinct_features)

        n_features_per_split, n_splits = self.__validate_split(n_distinct, n_common)

        splits = []

        for k in range(0, n_splits):
            i = k * n_features_per_split
            j = i + n_features_per_split

            indexes = np.append(common_features, distinct_features[i:j])
            splits.append(indexes)

        # Add left features
        total_splits = n_features_per_split * len(splits)
        n_left = n_distinct - total_splits

        if n_left > 0:
            splits[-1] += distinct_features[total_splits:total_splits + n_left]

        return splits

    def __set_random_state(self, random_state):
        self.__random_state = np.random.RandomState(random_state)

    def __get_n_features_per_split(self, n_distinct, n_splits):
        return math.floor(n_distinct / n_splits)

    def __validate_split(self, n_distinct, n_common):
        n_features_per_split = self.__get_n_features_per_split(n_distinct, self.n_splits)
        n_splits = self.n_splits

        # A split should contain more than 1 feature
        if n_features_per_split + n_common < 2:
            n_splits = n_distinct // (2 - n_common)
            n_features_per_split = self.__get_n_features_per_split(n_distinct, n_splits)

            self.__warn_new_n_splits(n_splits)

        return n_features_per_split, n_splits

    def __warn_new_n_splits(self, n_splits):
        warnings.warn('Each division has less than 2 features. \
                      Narrowing down to {} divisions.'.format(str(n_splits)), UserWarning)
