""""Data analysis using classification."""
from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv, concat, DataFrame
from .tree_analysis import TreeAnalysis
from .path import ClassificationPath
from glob import glob
from os import path
import numpy as np


class ClassificationAnalysis:

    def process(self, datasets_path, evaluation_path):
        """Create a dataset for each method (classifiers, agreggators).

        :param datasets_path: String
            Path to data sets' characteristics file.

        :param evaluation_path: String
            Path to evaluation results. Should contain folders with
            cv_summary.csv file.

        :return: None
        """
        score = 'f1'
        best_methods = self.__best_method_by_dataset([score, 'f1_micro'],
                                                     evaluation_path)

        self.__create_dataset_by_method(datasets_path, best_methods)

    @staticmethod
    def grow_trees():
        TreeAnalysis.grow_trees(DecisionTreeClassifier(), ClassificationPath())

    def __best_method_by_dataset(self, scores, evaluation_path):
        scores = [scores] if type(scores) is str else scores

        evaluation_paths = path.join(evaluation_path, '*')
        evaluation_folders = glob(evaluation_paths)

        best_method_by_dataset = {}

        for folder in evaluation_folders:
            if not path.isdir(folder):
                continue

            dataset_fullname = folder.split('/')[-1]
            dataset_metadata = dataset_fullname.split('_')

            dataset_name = dataset_metadata[0]
            dataset_overlap = int(dataset_metadata[-1]) / 10

            summary_path = path.join(folder, ClassificationPath().default_file)
            summary = read_csv(summary_path, header=[0, 1], index_col=0)

            try:
                summary = summary.sort_values(('mean', scores[0]))
            except KeyError:
                summary = summary.sort_values(('mean', scores[1]))

            best_method = summary.index.values[-1]

            best_method_by_dataset.setdefault(dataset_name, DataFrame())
            best_method_by_dataset[dataset_name] = \
                best_method_by_dataset[dataset_name].append([[dataset_overlap,
                                                             best_method]],
                                                            ignore_index=True)

        data = concat(best_method_by_dataset)
        data.columns = ['overlap', 'best_method']

        return data

    def __create_dataset_by_method(self, datasets_path, best_methods):
        methods = best_methods.index.values

        datasets_features = read_csv(datasets_path, header=0)
        datasets = datasets_features.loc[:, 'dataset']

        datasets_features.index = datasets
        datasets_features.drop('dataset', axis=1, inplace=True)

        datasets_features_cols = datasets_features.columns.values
        best_methods_cols = best_methods.columns.values

        cols = np.append(datasets_features_cols, best_methods_cols)
        data = DataFrame(columns=cols)

        for method, index in methods:
            dataset_ins = datasets_features.loc[method, :]
            best_methods_ins = best_methods.loc[method, :].iloc[index, :]

            ins_series = dataset_ins.append(best_methods_ins, ignore_index=True)
            instance = DataFrame(ins_series).T
            instance.columns = cols

            data = data.append(instance, ignore_index=True)

        data.to_csv(path.join(ClassificationPath().data_path,
                              'better_methods.csv'), index=False)
