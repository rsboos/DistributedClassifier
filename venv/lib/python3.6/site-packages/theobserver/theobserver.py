import numpy as np

from math import log2
from pandas import read_csv
from sklearn.metrics import silhouette_score


class Observer():
    """Observe and extract information about a dataset.

    Arguments
        filepath: string
            The dataset's absolute/relative path. Must be a CSV format file.

        target_i: {-1, 0}
            The target's column index. The default is -1.

    Return
        An Observer object
    """
    def __init__(self, filepath, target_i=-1):
        self.filepath = filepath
        self.target_i = target_i

    def __sep(self):
        file = open(self.filepath)
        line = file.readline()
        file.close()

        comma = line.count(',')
        semicolon = line.count(';')

        return ',' if comma > semicolon else ';'

    def __x(self):
        data = read_csv(self.filepath, self.__sep(), header=None)
        _, n_columns = data.shape

        # Initial and final column index (exclusive on stop)
        i = self.target_i + 1            # initial
        j = n_columns + self.target_i    # final + 1 (because it's exclusive)

        return data.iloc[:, i:j]

    def __y(self):
        data = read_csv(self.filepath, self.__sep(), header=None)
        return data.iloc[:, self.target_i]

    def n_instances(self):
        """Get the number of instances."""
        file = open(self.filepath)
        i = [1 for line in file]
        file.close()

        return sum(i)

    def n_features(self):
        """Get the number of features."""
        file = open(self.filepath)
        line = file.readline()
        file.close()

        d = self.__sep()

        return len(line.split(d)) - 1

    def n_targets(self):
        """Get the number of targets."""
        d = self.__sep()
        file = open(self.filepath)

        targets = [line.split(d)[self.target_i] for line in file]
        targets = set(targets)

        return len(targets)

    def silhouette(self):
        """Get the mean Silhouette Coefficient for all samples."""
        X = self.__x()
        y = self.__y()

        return silhouette_score(X, y)

    def entropy(self):
        """Get the samples' entropy."""
        y = self.__y()
        sety, counts = np.unique(y, return_counts=True)
        total = len(y)

        result = 0
        for target, n in zip(sety, counts):
            p = n / total
            result = result - (p * log2(p))

        return result

    def unbalanced(self):
        """Get the unbalaced metric, where 1 is very balanced and 0 extremely unbalaced."""
        sety = set(self.__y())
        n = len(sety)

        return self.entropy() / log2(n)

    def extract(self):
        """Extract all the observed information.

        Return
            [n_instances, n_features, n_targets, silhouette, unbalanced]
        """
        i = self.n_instances()
        f = self.n_features()
        t = self.n_targets()
        s = self.silhouette()
        u = self.unbalanced()

        return [i, f, t, s, u]

