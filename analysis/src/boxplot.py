import matplotlib.pyplot as plt
import numpy as np

from glob import glob
from pandas import read_csv

classifiers = ['gnb', 'dtree', 'svc', 'knn', 'mlp']
datasets_folders = glob('tests/*')
datasets = [d.split('_')[0].replace('tests/', '') for d in datasets_folders]
rankings = []

for folder in datasets_folders:
    data = read_csv(folder + '/cv_summary.csv', header=[0, 1], index_col=0)

    # Just mean accuracy
    try:
        data = data.loc[:, 'mean'].loc[:, 'f1_macro']
    except KeyError:
        data = data.loc[:, 'mean'].loc[:, 'f1']

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
    rankings.append(ranking)

m, n = len(rankings), len(rankings[0])
boxplot_data = np.reshape(rankings, (m, n))

fig, ax = plt.subplots()
ax.boxplot(boxplot_data)

ax.set_xlabel('Methods')
ax.set_ylabel('Rank Position')
ax.set_xticklabels(ordered_methods, rotation=90)

plt.yticks(range(n+1))
plt.show()
