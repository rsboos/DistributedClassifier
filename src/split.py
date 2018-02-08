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
