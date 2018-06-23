"""Create a dataset for each method (classifiers, agreggators)."""

import numpy as np
from pandas import read_csv, DataFrame


original = read_csv('../data/original.csv')
header = original.columns
original = original.values

X = original[:, :-1]
x_rows, x_cols = X.shape

datasets_names = original[:, -1]
n_rows = datasets_names.size


# Get f1_scores for all methods to all datasets
methods = np.array([])
f1_scores = {}
for n in range(n_rows):
    name = datasets_names[n]

    for i in range(0, 11):
        try:
            folder = '{}_last_{}'.format(name, i)
            filepath = '../../../tests/{}/cv_summary.csv'.format(folder)

            summary = read_csv(filepath, header=[0, 1], index_col=0)

            try:
                f1_score = summary.loc[:, 'mean'].loc[:, 'f1_micro']
            except KeyError:
                f1_score = summary.loc[:, 'mean'].loc[:, 'f1']

            f1_scores[folder] = f1_score

            prob_methods = f1_score.index.values
            if prob_methods.size > methods.size:
                methods = prob_methods

        except FileNotFoundError:
            print('{} dataset does not exist yet.'.format(folder))

# Create a data set for each method
for method in methods:
    data = None

    for n in range(n_rows):
        name = datasets_names[n]

        for i in range(0, 11):
            folder = '{}_last_{}'.format(name, i)

            if folder in f1_scores:
                ins = X[n, :]
                overlap = i / 10

                try:
                    method_f1 = f1_scores[folder].loc[method]

                    if method_f1 == 0:
                        ins = np.append(ins, overlap)
                        ins = np.append(ins, method_f1)

                        if data is None:
                            cols = ins.size
                            data = np.resize(ins, (1, cols))
                        else:
                            data = np.append(data, [ins], axis=0)
                except KeyError:
                    pass

    df = DataFrame(data)
    df.columns = list(header[:-1]) + ['overlap', 'f1_score']
    df.to_csv('data/{}_f1_scores.csv'.format(method), index=False)
