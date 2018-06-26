from sklearn.tree import DecisionTreeRegressor, export_graphviz
from pandas import read_csv, DataFrame
from glob import glob
import numpy as np
import os


def map_tree(tree_path, labels):
    tree = open(tree_path, 'r')
    mapped_tree = []

    for line in tree:
        one_line = line
        for i in range(labels.size):
            one_line = one_line.replace('X[{}]'.format(i), labels[i])

        mapped_tree.append(one_line)

    tree.close()

    new_tree = open(tree_path, 'w')

    for line in mapped_tree:
        new_tree.write(line)

    new_tree.close()

scored_ranks = {}
for dataset_path in glob('data/*'):

    data = read_csv(dataset_path)

    X = data.values[:, :-1]
    y = data.values[:, -1]

    model = DecisionTreeRegressor(max_depth=10, min_samples_split=0.1)
    model.fit(X, y)

    dataset = dataset_path.split('/')[-1]
    dataset = dataset.split('.')[0]
    tree_path = 'results/trees/' + dataset + '.dot'

    export_graphviz(model, tree_path)
    map_tree(tree_path, data.columns.values)
    os.system('dot -Tpng ' + tree_path + ' -o ' + tree_path[:-4] + '.png')

    feature_names = data.columns.values[:-1]
    feature_scores = model.feature_importances_

    scored_ranks[dataset] = [s for s in feature_scores]

scored_ranks_path = 'results/tree_ranks/scored_ranks.csv'
df = DataFrame(scored_ranks).T
df.columns = feature_names
df.T.to_csv(scored_ranks_path)


data = read_csv(scored_ranks_path, index_col=0, header=0)
methods = data.columns.values
features = data.index.values

m, n = data.shape

ranks = {}
for j in range(n):
    rank = data.iloc[:, j]

    k = methods[j]
    ranks[k] = []
    for i in reversed(np.argsort(rank)):
        if rank[i] == 0:
            ranks[k].append("")
        else:
            ranks[k].append(features[i])

df = DataFrame(ranks)
df = df.T.sort_index()
df.to_csv('results/tree_ranks/ranks.csv')
