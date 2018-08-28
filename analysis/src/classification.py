""""Data analysis using classification."""
import sys
sys.path.append('../evaluation/src/')

from pandas import read_csv, concat, DataFrame
from .tree_analysis import TreeAnalysis
from .path import ClassificationPath
from metrics import summary
from copy import deepcopy
from glob import glob
from os import path
import numpy as np
import os

from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import KFold, cross_validate

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class ClassificationAnalysis:

    def __init__(self):
        self.__method_dataset = "better_methods.csv"
        self.__method_type_dataset = "better_types.csv"

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

        best_types = self.__best_method_type_by_dataset(best_methods)

        self.__create_dataset_by_method(datasets_path, best_methods, self.__method_dataset)
        self.__create_dataset_by_method(datasets_path, best_types, self.__method_type_dataset)

    def evaluate(self, classifiers='*', scoring='*', cv=10, iterations=10):
        """Evaluate data in tests/classification/data/* and save results in
        tests/classification/results/evaluation.

        :param classifiers: dict or str (default '*')
            Classifiers to be used. When default, use all classifiers.

        :param scoring: dict or str (default '*')
            Metrics to be measured. Its values should be scorers.
            When default, use all metrics.

        :param cv: int (default 10)
            Number of CV folds.

        :param iterations: int (default 10)
            Number of CV repeats.

        :return: None
        """
        if classifiers == '*':
            classifiers = self.__default_classifiers()

        if scoring == '*':
            scoring = self.__default_scoring()

        dataset_paths = glob(path.join(ClassificationPath().data_path, '*'))

        for dataset_path in dataset_paths:

            scores = self.__cross_validate(dataset_path,
                                           classifiers,
                                           scoring,
                                           cv,
                                           iterations)

            dataset_file = dataset_path.split('/')[-1]
            dataset_name = dataset_file[:-4]

            scores_path = path.join(ClassificationPath().evaluation_path,
                                    dataset_name)

            if not path.exists(scores_path):
                os.makedirs(scores_path)

            for classifier, score in scores.items():
                filename = classifier + '.csv'
                filepath = path.join(scores_path, filename)
                score.to_csv(filepath, index=False)

            all_scores = list(scores.values())
            summary_path = path.join(scores_path,
                                     ClassificationPath().default_file)

            cv_summary = summary(all_scores)
            cv_summary.index = list(scores.keys())

            cv_summary.to_csv(summary_path)

    @staticmethod
    def grow_trees():
        TreeAnalysis.grow_trees(DecisionTreeClassifier(), ClassificationPath(), max_depth=5)

    @staticmethod
    def get_important_nodes(analysis_datapath):
        classification_path = ClassificationPath()
        TreeAnalysis.get_important_nodes(analysis_datapath, classification_path)

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

    def __best_method_type_by_dataset(self, best_methods):
        best_types = deepcopy(best_methods)
        m, _ = best_types.shape

        for i in range(m):
            method_type = ClassificationPath.concat_method_type(best_types.iloc[i, -1])
            best_types.iloc[i, -1] = method_type.split('_')[0]

        return best_types

    def __create_dataset_by_method(self, datasets_path, best_methods, output):
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
                              output), index=False)

    def __default_classifiers(self):
        return {"gnb": GaussianNB(),
                "svc": SVC(probability=True),
                "mlp": MLPClassifier(),
                "dtree": DecisionTreeClassifier(),
                "knn": KNeighborsClassifier()}

    def __default_scoring(self):
        return {"f1_macro": make_scorer(f1_score, average='macro'),
                "f1_micro": make_scorer(f1_score, average='micro'),
                "recall_macro": make_scorer(recall_score, average='macro'),
                "recall_micro": make_scorer(recall_score, average='micro'),
                "accuracy": make_scorer(accuracy_score),
                "precision_macro": make_scorer(precision_score, average='macro'),
                "precision_micro": make_scorer(precision_score, average='micro')}

    @staticmethod
    def __cross_validate(dataset_path, classifiers, scoring, cv, iterations):
        data = read_csv(dataset_path, header=0)
        data = data.values

        X = data[:, :-1]
        y = data[:, -1]

        scores = {}

        for i in range(iterations):
            skf = KFold(cv, True, i)

            for name, classifier in classifiers.items():
                folds = skf.split(X, y)
                scores.setdefault(name, DataFrame())

                cv_scores = cross_validate(classifier, X, y,
                                           cv=folds,
                                           scoring=scoring)

                cv_scores = {k.replace('test_', ''): v for k, v in
                             cv_scores.items()
                             if 'test_' in k}

                cv_scores = DataFrame(cv_scores)
                scores[name] = scores[name].append(cv_scores)

        return scores
