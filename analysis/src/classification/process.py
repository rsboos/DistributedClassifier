import numpy as np
from pandas import read_csv, DataFrame


original = read_csv('../data/datasets.csv')
header = original.columns
original = original.values

X = original[:, :-1]
x_rows, x_cols = X.shape

datasets_names = original[:, -1]
n_rows = datasets_names.size

# Get f1_scores for all methods to all datasets
better_methods = {}

for n in range(n_rows):
    name = datasets_names[n]
    better_methods[name] = []

    for i in range(0, 11):
        try:
            folder = '{}_last_{}'.format(name, i)
            filepath = '../../../tests/{}/cv_summary.csv'.format(folder)

            summary = read_csv(filepath, header=[0, 1], index_col=0)

            try:
                f1_score = summary.loc[:, 'mean'].loc[:, 'f1_macro']
            except KeyError:
                f1_score = summary.loc[:, 'mean'].loc[:, 'f1']

            better_methods[name].append((i, f1_score.argmax()))

        except FileNotFoundError:
            print('{} dataset does not exist yet.'.format(folder))

# Create a dataset for each method
data = None

for n in range(n_rows):
    name = datasets_names[n]

    if name in better_methods:
        for i, method in better_methods[name]:
            ins = X[n, :]
            overlap = i / 10

            ins = np.append(ins, overlap)
            ins = np.append(ins, method)

            if data is None:
                cols = ins.size
                data = np.resize(ins, (1, cols))
            else:
                data = np.append(data, [ins], axis=0)

df = DataFrame(data)
df.columns = list(header[:-1]) + ['overlap', 'better_method']
df.to_csv('data/better_methods.csv', index=False)
