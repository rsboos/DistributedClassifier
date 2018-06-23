from sklearn.tree import DecisionTreeRegressor, export_graphviz
from pandas import read_csv, DataFrame, concat
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
ranks = {}
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
    scored_ranks[dataset] = model.feature_importances_
    rank = [feature_names[i] for i in np.argsort(scored_ranks[dataset])]
    ranks[dataset] = list(reversed(rank))

df = DataFrame(scored_ranks).T
df.columns = feature_names
df.T.to_csv('results/tree_ranks/scored_ranks.csv')

df = DataFrame(ranks)
df = df.T.sort_index()
df.to_csv('results/tree_ranks/ranks.csv')
