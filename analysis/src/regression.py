""""Data analysis using regression."""
import sys
sys.path.append('../evaluation/src/')

import os
import numpy as np
from os import path
from glob import glob
from joblib import dump
from copy import deepcopy
from metrics import summary
from itertools import product
from pandas import read_csv, DataFrame, concat, Series
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
        classes = {'arbmd', 'arbmdi', 'arbmdic', 'classif', 'cmb', 'math', 'scf', 'vote'}
        best_regressors_path = path.join(r_path.analysis_path, 'best_regressors_cleaned.csv')
        best_regressors = read_csv(best_regressors_path, header=[0], index_col=[0])

        if regressors == '*':
            regressors = self.__default_regressors()

        regressors_mdl = {}

        # Train best regressors
        for aggr in best_regressors.index.values:
            regressor_name = best_regressors.loc[aggr, 'best_regressor']
            mdl = deepcopy(regressors[regressor_name])

            data_filename = aggr + '_f1.csv'
            data_filepath = path.join(r_path.data_path, data_filename)
            data = read_csv(data_filepath, header=[0], index_col=None)
            x, y = data.iloc[:, :-1], data.iloc[:, -1]

            mdl.fit(x, y)
            regressors_mdl[aggr] = mdl

            # dump(mdl, path.join(r_path.regressors_path, aggr + '.joblib'))

        # Create ranks for each dataset
        test_data_path = path.join(evaluation_path, '../datasets_tests')
        datasets_folders = glob(path.join(test_data_path, '*_0'))
        datasets_info = read_csv('data/datasets.csv', header=[0], index_col=[-1])

        for folder in datasets_folders:
            dataset_name = folder.split('/')[-1][:-2]
            summary_path = path.join(folder, 'cv_summary.csv')
            summary = read_csv(summary_path, header=[0, 1], index_col=[0])

            index_values = [Path.fix_method_name(i) for i in summary.index.values]
            summary = DataFrame(summary.loc[:, 'mean'].to_dict('list'), index=index_values)

            try:
                summary = summary.loc[:, scores[0]]
            except KeyError:
                summary = summary.loc[:, scores[1]]

            summary = summary.sort_values(ascending=False)
            summary = summary.map(lambda a: round(a, 2))

            rank_path = path.join(r_path.analysis_path, '{}_true_rank.csv'.format(dataset_name))
            summary.to_csv(rank_path, header=False, index=True)

            dataset_info = datasets_info.loc[dataset_name, :].values
            rank = {}
            for aggr in regressors_mdl:
                if aggr not in classes:
                    rank[aggr] = regressors_mdl[aggr].predict([dataset_info])[0]

            rank = Series(list(rank.values()), index=list(rank.keys()))
            rank.sort_values(inplace=True, ascending=False)

            rank = rank.map(lambda a: round(a, 2))

            rank_path = path.join(r_path.analysis_path, '{}_predicted_rank.csv'.format(dataset_name))
            rank.to_csv(rank_path, header=False, index=True)

    def compare_ranks(self, evaluation_path):
        r_path = RegressionPath()
        test_data_path = path.join(evaluation_path, '../datasets_tests')
        datasets = [folder.split('/')[-1][:-2] for folder in glob(path.join(test_data_path, '*_0'))]

        readable_dt_name = {'sky_last': 'Sloan digital sky',
                            'emg_last': 'Gesture',
                            'plates_last': 'Steel plates’ fault',
                            'theorem_last': 'First order theorem',
                            'lifeexpectancy_last': 'Life expectancy',
                            'credit_last': 'Credit',
                            'pulsar_last': 'Pulsar star',
                            'politics_last': 'Turkey political opinions',
                            'speech_last': 'Speech',
                            'income_last': 'Income'}

        data = []
        for dataset_name in datasets:
            true_rank_path = path.join(r_path.analysis_path, '{}_true_rank.csv'.format(dataset_name))
            pred_rank_path = path.join(r_path.analysis_path, '{}_predicted_rank.csv'.format(dataset_name))

            true_rank = read_csv(true_rank_path, header=None, index_col=None)
            pred_rank = read_csv(pred_rank_path, header=None, index_col=None)

            score_rank, score_bucket, buckets_count = self.__kendall_tau(true_rank, pred_rank)

            data.append((readable_dt_name[dataset_name],
                         score_rank,
                         score_bucket,
                         buckets_count[0],
                         buckets_count[1],
                         buckets_count[2],
                         buckets_count[3]))

        results_path = path.join(r_path.analysis_path, 'ranks_scores.csv')
        with open(results_path, 'w') as file:
            file.write('dataset,score_rank,score_bucket,1,2,3,4\n')
            for dt, score_rank, score_bucket, b1, b2, b3, b4 in data:
                file.write("{},{},{},{},{},{},{}\n".format(dt, score_rank, score_bucket, b1, b2, b3, b4))

        _, _, _, b1, b2, b3, b4 = zip(*data)
        b1_mean, b2_mean, b3_mean, b4_mean = np.mean(b1), np.mean(b2), np.mean(b3), np.mean(b4)
        b1_std, b2_std, b3_std, b4_std = np.std(b1), np.std(b2), np.std(b3), np.std(b4)

        results_path = path.join(r_path.analysis_path, 'buckets_results.csv')
        with open(results_path, 'w') as file:
            file.write('mean,std\n')
            for b_mean, b_std in [(b1_mean, b1_std), (b2_mean, b2_std), (b3_mean, b3_std), (b4_mean, b4_std)]:
                file.write("{},{}\n".format(b_mean, b_std))

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

    def __get_buckets(self, data):
        buckets = {(0.00, 0.50): [],
                   (0.50, 0.75): [],
                   (0.75, 0.90): [],
                   (0.90, 1.01): []}

        for i in range(data.shape[0]):
            f1 = data.iloc[i, 1]
            f1 = 1 if f1 > 1 else f1

            for drange in buckets:
                if drange[0] <= f1 < drange[1]:
                    buckets[drange].append(data.iloc[i, 0])
                    break

        return buckets

    def __find_bucket(self, buckets, element):
        n = len(buckets)

        for i in range(n):
            if element in buckets[i]:
                return i

        return None

    def __pos(self, buckets, i):
        lens = [0]
        for j in range(i):
            lens.append(len(buckets[j]))

        return sum(lens) + (len(buckets[i]) + 1) / 2

    def __kendall_tau(self, rank1, rank2):
        buckets1 = self.__get_buckets(rank1)
        buckets2 = self.__get_buckets(rank2)

        buckets1_values = list(buckets1.values())
        buckets2_values = list(buckets2.values())

        buckets1_intervals = list(buckets1.keys())

        rank1_items = list(rank1.iloc[:, 0].values)
        rank2_items = list(rank2.iloc[:, 0].values)

        rank1_set = set(rank1_items)
        rank2_set = set(rank2_items)

        if len(rank1_items) > len(rank2_items):
            diff_values = rank1_set - rank2_set
            rank1_items = list(rank1_set - diff_values)
        elif len(rank1_items) < len(rank2_items):
            diff_values = rank2_set - rank1_set
            rank2_items = list(rank2_set - diff_values)

        p = [(i, j) for i, j in product(rank1_items, rank2_items) if i != j]

        penalties_rank = []
        penalties_bucket = []
        buckets_count = [0] * len(buckets1_intervals)

        for i, j in p:
            penalty_rank = self.__penalty_with_ranks(i, j, rank1_items, rank2_items)
            penalty_bucket, b1_i, b1_j, b2_i, b2_j = self.__penalty_with_buckets(i, j, buckets1_values, buckets2_values)

            penalties_rank.append(penalty_rank)
            penalties_bucket.append(penalty_bucket)

            if penalty_bucket > 0:
                buckets_count[b1_i] += 1

                if b1_j != b1_i:
                    buckets_count[b1_j] += 1

                if b2_i != b1_i and b2_i != b1_j:
                    buckets_count[b2_i] += 1

                if b2_j != b1_i and b2_j != b1_j and b2_j != b2_i:
                    buckets_count[b2_j] += 1

        return sum(penalties_rank) / len(p), sum(penalties_bucket) / len(p), np.array(buckets_count) / len(p)

    def __penalty_with_buckets(self, i, j, buckets1, buckets2):
        b1_i = self.__find_bucket(buckets1, i)
        b1_j = self.__find_bucket(buckets1, j)

        b2_i = self.__find_bucket(buckets2, i)
        b2_j = self.__find_bucket(buckets2, j)

        pos1_i = self.__pos(buckets1, b1_i)
        pos1_j = self.__pos(buckets1, b1_j)

        pos2_i = self.__pos(buckets2, b2_i)
        pos2_j = self.__pos(buckets2, b2_j)

        penalty = 0

        # Case 1
        # # Same order
        if (pos1_i > pos1_j and pos2_i > pos2_j) or (pos1_i < pos1_j and pos2_i < pos2_j):
            penalty = 0

        # # Opposite order
        elif (pos1_i > pos1_j and pos2_i < pos2_j) or (pos1_i < pos1_j and pos2_i > pos2_j):
            penalty = 1

        # Case 2
        # # i and j are in the same bucket (tied)
        elif b1_i == b1_j and b2_i == b2_j:
            penalty = 0

        # Case 3
        # # i and j are in the same bucket in just one rank
        elif (b1_i == b1_j and b2_i != b2_j) or (b1_i != b1_j and b2_i == b2_j):
            penalty = 1 / 2

        return penalty, b1_i, b1_j, b2_i, b2_j

    def __penalty_with_ranks(self, i, j, rank1, rank2):
        x_i = rank1.index(i)
        y_i = rank2.index(i)

        x_j = rank1.index(j)
        y_j = rank2.index(j)

        # concordantes
        if (x_i > x_j and y_i > y_j) or (x_i < x_j and y_i < y_j):
            return 0

        # discordantes
        if (x_i > x_j and y_i < y_j) or (x_i < x_j and y_i > y_j):
            return 1

        return 0
