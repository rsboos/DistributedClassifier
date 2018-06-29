""""Data analysis using regression."""

import sys
sys.path.append('../evaluation/src/')

import os
import numpy as np
from os import path
from glob import glob
from pandas import read_csv, DataFrame, concat
from metrics import summary
from sklearn.model_selection import KFold, cross_validate

from sklearn.metrics import explained_variance_score, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score, make_scorer

from sklearn.linear_model import RANSACRegressor, HuberRegressor, \
    TheilSenRegressor, BayesianRidge, OrthogonalMatchingPursuit, ElasticNet, \
    LinearRegression, ARDRegression, LassoLars, Lasso, Ridge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge


class RegressionAnalysis():
    default_file = 'cv_summary.csv'
    test_path = 'tests/regression'
    data_path = path.join(test_path, 'data/')
    results_path = path.join(test_path, 'results/')
    evaluation_path = path.join(results_path, 'evaluation/')

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
        f1_scores = self.__scores_by_method([score, 'f1_micro'], evaluation_path)

        self.__create_dataset_by_method(datasets_path, f1_scores, score)
        self.__remove_zeroed_instances()

    def evaluate(self, regressors='*', scoring='*', cv=10, iterations=10):
        """Evaluate data in tests/regression/data/* and save results in
        tests/regression/results/evaluation.

        :param regressors: dict or str (default '*')
            Regressors to be used. When default, use all regressors.

        :param scoring: dict or str (default '*')
            Metrics to be measured. Its values should be scorers.
            When default, use all metrics.

        :param cv: int (default 10)
            Number of CV folds.

        :param iterations: int (default 10)
            Number of CV repeats.

        :return: None
        """
        if regressors == '*':
            regressors = self.__default_regressors()

        if scoring == '*':
            scoring = self.__default_scoring()

        dataset_paths = glob(path.join(self.data_path, '*'))

        for dataset_path in dataset_paths:

            scores = self.__cross_validate(dataset_path,
                                           regressors,
                                           scoring,
                                           cv,
                                           iterations)

            dataset_file = dataset_path.split('/')[-1]
            dataset_name = dataset_file[:-4]

            scores_path = path.join(self.evaluation_path, dataset_name)
            if not path.exists(scores_path):
                os.makedirs(scores_path)

            for regressor, score in scores.items():
                filename = regressor + '.csv'
                filepath = path.join(scores_path, filename)
                score.to_csv(filepath, index=False)

            all_scores = list(scores.values())
            summary_path = path.join(scores_path, self.default_file)

            cv_summary = summary(all_scores)
            cv_summary.index = list(scores.keys())

            cv_summary.to_csv(summary_path)


    def __create_dataset_by_method(self, datasets_path, f1_scores, score):
        methods = f1_scores.columns.values

        datasets_features = read_csv(datasets_path, header=0)
        datasets = datasets_features.loc[:, 'dataset']

        datasets_features.index = datasets
        datasets_features.drop('dataset', axis=1, inplace=True)

        for method in methods:
            method_f1 = f1_scores.loc[:, method]
            data = DataFrame()

            datasets, _ = zip(*method_f1.index.values)
            datasets = set(datasets)

            for dataset in datasets:
                dataset_instance = DataFrame(
                    datasets_features.loc[dataset, :]).T

                dataset_f1 = DataFrame(method_f1.loc[dataset, :])
                dataset_f1.columns = [score]

                n = dataset_f1.shape[0]
                dataset_instance = concat([dataset_instance] * n)

                _, overlaps = zip(*dataset_f1.index.values)
                overlaps = np.array(overlaps)

                dataset_f1.index = overlaps
                dataset_instance.index = overlaps

                overlaps = DataFrame({'overlap': overlaps / 10})

                instances = concat([dataset_instance, overlaps, dataset_f1],
                                   axis=1)
                data = data.append(instances)

            data.to_csv(path.join(self.data_path,
                                  '{}_f1.csv'.format(method)),
                        index=False)

    def __scores_by_method(self, scores, evaluation_path):
        scores = [scores] if type(scores) is str else scores

        evaluation_paths = path.join(evaluation_path, '*')
        evaluation_folders = glob(evaluation_paths)

        f1_scores_by_dataset = {}

        for folder in evaluation_folders:
            if not path.isdir(folder):
                continue

            dataset_fullname = folder.split('/')[-1]
            dataset_metadata = dataset_fullname.split('_')

            dataset_name = dataset_metadata[0]
            dataset_overlap = int(dataset_metadata[-1])

            summary_path = path.join(folder, self.default_file)
            summary = read_csv(summary_path, header=[0, 1], index_col=0)
            summary = summary.sort_index()

            try:
                summary_f1 = summary.loc[:, 'mean'].loc[:, scores[0]]
            except KeyError:
                summary_f1 = summary.loc[:, 'mean'].loc[:, scores[1]]

            summary_f1 = DataFrame(summary_f1).T
            summary_f1.index = [dataset_overlap]

            f1_scores_by_dataset[dataset_name] = \
                f1_scores_by_dataset.setdefault(dataset_name,
                                                DataFrame()).append(summary_f1)

        frames = f1_scores_by_dataset.values()
        keys = f1_scores_by_dataset.keys()
        f1_scores = concat(frames, keys=keys, copy=False)

        return f1_scores.sort_index()

    def __remove_zeroed_instances(self):
        data_files = glob(path.join(self.data_path, '*'))
        removed_instances = []

        for data_file in data_files:
            data = read_csv(data_file, header=0)
            scores = data.iloc[:, -1].values

            x = np.where(scores == 0)
            removed_instances += list(x[0])

        removed_instances = set(removed_instances)

        for data_file in data_files:
            data = read_csv(data_file, header=0)
            data = data.drop(removed_instances)
            data.to_csv(data_file, index=False)

    @staticmethod
    def __default_regressors():
        return {'ransac': RANSACRegressor(),
                'huber': HuberRegressor(),
                'theil_sen': TheilSenRegressor(),
                'linear': LinearRegression(),
                'ard': ARDRegression(),
                'orthogonal_matching': OrthogonalMatchingPursuit(),
                'elastic_net': ElasticNet(),
                'bayesian_ridge': BayesianRidge(),
                'lasso_lars': LassoLars(),
                'lasso': Lasso(),
                'ridge': Ridge(),
                'gaussian_process': GaussianProcessRegressor(),
                'decision_tree': DecisionTreeRegressor(),
                'svr': SVR(),
                'nu_svr': NuSVR(),
                'linear_svr': LinearSVR(),
                'mlp': MLPRegressor(),
                'kernel_ridge': KernelRidge()}

    @staticmethod
    def __default_scoring():
        return {'explained_variance': make_scorer(explained_variance_score),
                'mean_absolute': make_scorer(mean_absolute_error),
                'mean_square': make_scorer(mean_squared_error),
                'median_absolute': make_scorer(median_absolute_error),
                'r2': make_scorer(r2_score)}

    def __cross_validate(self, dataset_path, regressors, scoring, cv, iterations):
        data = read_csv(dataset_path, header=0)
        data = data.values

        X = data[:, :-1]
        y = data[:, -1]

        scores = {}

        for i in range(iterations):
            skf = KFold(cv, True, i)

            for name, regressor in regressors.items():
                folds = skf.split(X, y)
                scores.setdefault(name, DataFrame())

                cv_scores = cross_validate(regressor, X, y, cv=folds,
                                           scoring=scoring)

                cv_scores = {k.replace('test_', ''): v for k, v in
                             cv_scores.items()
                             if 'test_' in k}

                cv_scores = DataFrame(cv_scores)
                scores[name] = scores[name].append(cv_scores)

        return scores