from pandas import read_csv, DataFrame, concat
from glob import glob
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import make_scorer

from sklearn.linear_model import RANSACRegressor, HuberRegressor, TheilSenRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit, ElasticNet, BayesianRidge
from sklearn.linear_model import LinearRegression, ARDRegression
from sklearn.linear_model import LassoLars, Lasso, Ridge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge


regressors = {'ransac': RANSACRegressor(),
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

scoring = {'explained_variance': make_scorer(explained_variance_score),
           'mean_absolute': make_scorer(mean_absolute_error),
           'mean_square': make_scorer(mean_squared_error),
           'median_absolute': make_scorer(median_absolute_error),
           'r2': make_scorer(r2_score)}

for dataset_path in glob('data/*'):

    data = read_csv(dataset_path)
    data = data.values

    X = data[:, :-1]
    y = data[:, -1]

    for i in range(10):
        skf = KFold(10, True, i)

        scores = {'mean': {}, 'std': {}}
        for name, regressor in regressors.items():
            folds = skf.split(X, y)

            scores['mean'].setdefault(name, {})
            scores['std'].setdefault(name, {})

            try:
                cv_scores = cross_validate(regressor, X, y, cv=folds, scoring=scoring)
            except np.linalg.linalg.LinAlgError:
                continue

            for metric in scoring.keys():
                metric_score = cv_scores['test_' + metric]

                scores['mean'][name].setdefault(metric, [])
                scores['std'][name].setdefault(metric, [])

                scores['mean'][name][metric].append(metric_score.mean())
                scores['std'][name][metric].append(metric_score.std())

    for k in scores.keys():
        for regressor in scores[k].keys():
            for metric in scores[k][regressor].keys():
                if k == 'mean':
                    scores[k][regressor][metric] = np.mean(scores[k][regressor][metric])
                else:
                    scores[k][regressor][metric] = np.std(scores[k][regressor][metric])

    dataset_name = '_'.join(dataset_path.split('/')[-1].split('_')[:-2])

    mean = DataFrame(scores['mean']).T
    std = DataFrame(scores['std']).T
    df = concat([mean, std], keys=['mean', 'std'], axis=1, copy=False)

    df.to_csv('results/evaluation/' + dataset_name + '.csv')
