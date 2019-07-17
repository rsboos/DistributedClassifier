""""Data analysis using regression."""
import sys
sys.path.append('../evaluation/src/')

import os
import numpy as np
from os import path
from glob import glob
from copy import deepcopy
from metrics import summary
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import KFold, cross_validate

from sklearn.metrics import explained_variance_score, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score, make_scorer

from sklearn.linear_model import HuberRegressor, \
    TheilSenRegressor, BayesianRidge, OrthogonalMatchingPursuit, ElasticNet, \
    LinearRegression, ARDRegression, LassoLars, Lasso, Ridge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from .tree_analysis import TreeAnalysis
from .path import RegressionPath, Path
from sklearn.svm import SVR, NuSVR


class RegressionAnalysis:

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
        f1_scores = self.__scores_by_method([score, 'f1_micro'],
                                            evaluation_path)

        self.__create_dataset_by_method(datasets_path, f1_scores, score)
        self.__remove_zeroed_instances()
        self.__remove_nan()

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

        dataset_paths = glob(path.join(RegressionPath().data_path, '*'))

        for dataset_path in dataset_paths:

            scores = self.__cross_validate(dataset_path,
                                           regressors,
                                           scoring,
                                           cv,
                                           iterations)

            dataset_file = dataset_path.split('/')[-1]
            dataset_name = dataset_file[:-4]

            scores_path = path.join(RegressionPath().evaluation_path, dataset_name)
            if not path.exists(scores_path):
                os.makedirs(scores_path)

            for regressor, score in scores.items():
                filename = regressor + '.csv'
                filepath = path.join(scores_path, filename)
                score.to_csv(filepath, index=False)

            all_scores = list(scores.values())
            summary_path = path.join(scores_path, RegressionPath().default_file)

            cv_summary = summary(all_scores)
            cv_summary.index = list(scores.keys())

            cv_summary.to_csv(summary_path)

    def analyse(self):
        r_path = RegressionPath()
        evaluation_path = r_path.evaluation_path
        folders = glob(path.join(evaluation_path, '*'))
        summary = {}

        for aggr in folders:
            file = path.join(aggr, 'cv_summary.csv')
            aggr_name = Path.fix_method_name(aggr)

            results = read_csv(file, header=[0, 1], index_col=0)

            mean_values = results.loc[:, 'mean'].loc[:, 'mean_square']
            std_values = results.loc[:, 'std'].loc[:, 'mean_square']

            best_regression_in_mean = mean_values.idxmin()
            best_regression_in_std = std_values.idxmin()

            mse_mean_of_best_regressor_in_mean = mean_values.loc[best_regression_in_mean]
            mse_std_of_best_regressor_in_mean = std_values.loc[best_regression_in_mean]

            mse_mean_of_best_regressor_in_std = mean_values.loc[best_regression_in_std]
            mse_std_of_best_regressor_in_std = std_values.loc[best_regression_in_std]

            summary.setdefault('aggregation_method', [])
            summary['aggregation_method'].append(aggr_name)

            summary.setdefault('best_regressor_in_mean', [])
            summary['best_regressor_in_mean'].append(best_regression_in_mean)

            summary.setdefault('best_regressor_in_std', [])
            summary['best_regressor_in_std'].append(best_regression_in_std)

            summary.setdefault('mse_mean_of_best_regressor_in_mean', [])
            summary['mse_mean_of_best_regressor_in_mean'].append(mse_mean_of_best_regressor_in_mean)

            summary.setdefault('mse_std_of_best_regressor_in_mean', [])
            summary['mse_std_of_best_regressor_in_mean'].append(mse_std_of_best_regressor_in_mean)

            summary.setdefault('mse_mean_of_best_regressor_in_std', [])
            summary['mse_mean_of_best_regressor_in_std'].append(mse_mean_of_best_regressor_in_std)

            summary.setdefault('mse_std_of_best_regressor_in_std', [])
            summary['mse_std_of_best_regressor_in_std'].append(mse_std_of_best_regressor_in_std)

        df = DataFrame(summary)
        df.to_csv(path.join(r_path.analysis_path, 'best_regressors.csv'), header=True, index=False)

    def rank(self, evaluation_path, scores, regressors='*'):
        r_path = RegressionPath()
        best_regressors_path = path.join(r_path.analysis_path, 'best_regressors_cleaned.csv')
        best_regressors = read_csv(best_regressors_path, header=[0], index_col=[0])

        if regressors == '*':
            regressors = self.__default_regressors()

        regressors_mdl = {}

        # Train best regressors
        for aggr in best_regressors.indexes.values:
            regressor_name = best_regressors.loc[aggr, 'best_regressor']
            mdl = deepcopy(regressors[regressor_name])

            data_filename = aggr + '_f1.csv'
            data_filepath = path.join(r_path.data_path, data_filename)
            data = read_csv(data_filepath, header=[0], index_col=None)
            x, y = data.iloc[:, :-1], data.iloc[:, -1]

            mdl.fit(x, y)
            regressors_mdl[aggr] = mdl

        # Create ranks for each dataset
        test_data_path = path.join(evaluation_path, '../datasets_test')
        datasets_folders = glob(path.join(test_data_path, '*_0'))
        datasets_info = read_csv('data/datasets.csv', header=[0], index_col=[-1])

        for folder in datasets_folders:
            dataset_name = folder.split('/')[-1][:-2]
            summary_path = path.join(folder, 'cv_summary.csv')
            summary = read_csv(summary_path, header=[0, 1], index_col=[0])

            try:
                summary = summary.loc[:, 'mean'].loc[:, scores[0]]
                score = scores[0]
            except KeyError:
                summary = summary.loc[:, 'mean'].loc[:, scores[1]]
                score = scores[1]

            summary.sort_values(score, inplace=True, ascending=False)

            rank_path = path.join(r_path.analysis_path, '{}_true_rank.csv'.format(dataset_name))
            summary.to_csv(rank_path, header=True, index=True)

            dataset_info = datasets_info.loc[dataset_name, :].values
            rank = {}
            for aggr in regressors_mdl:
                rank[aggr] = regressors_mdl[aggr].predict([dataset_info])[0]

            rank = DataFrame(rank)
            rank = rank.T
            rank.columns.values = ['', score]

            rank_path = path.join(r_path.analysis_path, '{}_predicted_rank.csv'.format(dataset_name))
            rank.to_csv(rank_path, header=True, index=True)

    @staticmethod
    def grow_trees():
        TreeAnalysis.grow_trees(DecisionTreeRegressor(), RegressionPath())

    @staticmethod
    def get_important_nodes(analysis_datapath):
        TreeAnalysis.get_important_nodes(analysis_datapath, RegressionPath())

    def __create_dataset_by_method(self, datasets_path, f1_scores, score):
        methods = f1_scores.columns.values

        datasets_features = read_csv(datasets_path, header=0)
        datasets = datasets_features.loc[:, 'Dataset']

        datasets_features.index = datasets
        datasets_features.drop('Dataset', axis=1, inplace=True)

        data_methods_type = {}

        for method in methods:
            method_f1 = f1_scores.loc[:, method]

            method_type = RegressionPath.concat_method_type(method).split('_')[0]
            data_methods_type.setdefault(method_type, DataFrame())

            data = DataFrame()

            datasets, _ = zip(*method_f1.index.values)
            datasets = set(datasets)

            for dataset in datasets:
                dataset_instance = DataFrame(datasets_features.loc[dataset, :]).T
                dataset_instance.index = [(dataset_instance.index.values[0], 0)]

                dataset_f1 = DataFrame(method_f1.loc[dataset, :])
                dataset_f1.columns = [score]

                instances = concat([dataset_instance, dataset_f1], axis=1)

                data = data.append(instances)
                data_methods_type[method_type] = data_methods_type[method_type].append(instances)

            data.columns = list(map(lambda x: x.capitalize().replace('_', ' '), data.columns.values))
            data.to_csv(path.join(RegressionPath().data_path, '{}_f1.csv'.format(method)), index=False)

        for method_type in data_methods_type:
            data = data_methods_type[method_type]
            data.columns = list(map(lambda x: x.capitalize().replace('_', ' '), data.columns.values))
            data.to_csv(path.join(RegressionPath().data_path, '{}_f1.csv'.format(method_type)), index=False)

    def __scores_by_method(self, scores, evaluation_path):
        scores = [scores] if type(scores) is str else scores

        evaluation_paths = path.join(evaluation_path, '*_0*')
        evaluation_folders = glob(evaluation_paths)

        f1_scores_by_dataset = {}

        for folder in evaluation_folders:
            if not path.isdir(folder):
                continue

            dataset_fullname = folder.split('/')[-1]
            dataset_metadata = dataset_fullname.split('_')

            dataset_name = dataset_metadata[0]
            dataset_overlap = int(dataset_metadata[-1])

            summary_path = path.join(folder, RegressionPath().default_file)
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
        data_files = glob(path.join(RegressionPath().data_path, '*'))
        removed_instances = []

        for data_file in data_files:
            data = read_csv(data_file, header=0)
            scores = data.iloc[:, -1].values

            x = np.where(scores == 0)
            removed_instances += list(x[0])

        removed_instances = list(set(removed_instances))

        for data_file in data_files:
            data = read_csv(data_file, header=0)
            n_lines = data.shape[0]
            remove = [i for i in range(n_lines) if i in removed_instances]
            data = data.drop(remove)
            data.to_csv(data_file, index=False)

    def __remove_nan(self):
        data_files = glob(path.join(RegressionPath().data_path, '*'))

        for data_file in data_files:
            data = read_csv(data_file, header=0)
            data = data.dropna()
            data.to_csv(data_file, index=False)

    @staticmethod
    def __default_regressors():
        return {'huber': HuberRegressor(),
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
                'kernel_ridge': KernelRidge()}

    @staticmethod
    def __default_scoring():
        return {'explained_variance': make_scorer(explained_variance_score),
                'mean_absolute': make_scorer(mean_absolute_error),
                'mean_square': make_scorer(mean_squared_error),
                'median_absolute': make_scorer(median_absolute_error),
                'r2': make_scorer(r2_score)}

    @staticmethod
    def __cross_validate(dataset_path, regressors, scoring, cv, iterations):
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

                try:
                    cv_scores = cross_validate(regressor, X, y,
                                               cv=folds,
                                               scoring=scoring)

                    cv_scores = {k.replace('test_', ''): v for k, v in
                                 cv_scores.items()
                                 if 'test_' in k}

                    cv_scores = DataFrame(cv_scores)
                    scores[name] = scores[name].append(cv_scores)
                except np.linalg.linalg.LinAlgError:
                    pass

        return scores
