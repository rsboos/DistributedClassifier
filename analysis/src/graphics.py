import numpy as np
import matplotlib.pyplot as plt

from os import path
from glob import glob
from pandas import read_csv
from .path import RegressionPath


class Boxplot:

    def __init__(self, metric='f1_macro'):
        self.rankings = []
        self.ordered_methods = []
        self.__getter_data(metric)

    def show(self):
        """Show boxplot."""
        self.__make()
        plt.show()

    def save(self, filename):
        """Save figure in Path.graphics_path.

        :param filename: String
            File name.

        :return: None
        """
        self.__make()
        plt.savefig(path.join(RegressionPath().graphics_path, filename), bbox_inches='tight')

    def __getter_data(self, metric):
        type_path = RegressionPath()
        metric = [metric, 'f1'] if 'f1_' in metric else [metric, metric]
        classifiers = ['gnb', 'dtree', 'svc', 'knn', 'mlp']
        tests_path = '../evaluation/tests/'

        datasets_folders = [p for p in glob(path.join(tests_path, '*')) if path.isdir(p)]

        for folder in datasets_folders:
            data = read_csv(path.join(folder, type_path.default_file),
                            header=[0, 1], index_col=0)

            try:
                data = data.loc[:, 'mean'].loc[:, metric[0]]
            except KeyError:
                data = data.loc[:, 'mean'].loc[:, metric[1]]

            data = data.sort_values()
            methods = list(data.index.values)

            i = 0
            while len(methods) < 32:
                methods.append(classifiers[i])
                i += 1

            positions = list(enumerate(methods))

            # Sort by name
            positions.sort(key=lambda x: x[1])
            ranking, ordered_methods = zip(*positions)

            # From 1 to 32
            ranking = list(map(lambda x: x + 1, ranking))
            self.rankings.append(ranking)

        self.ordered_methods = ordered_methods

    def __make(self):
        m, n = len(self.rankings), len(self.rankings[0])
        boxplot_data = np.reshape(self.rankings, (m, n))

        fig, ax = plt.subplots()
        ax.boxplot(boxplot_data)

        ax.set_xlabel('Methods')
        ax.set_ylabel('Rank Position')
        ax.set_xticklabels(self.ordered_methods, rotation=90)
        plt.yticks(range(n + 1))
