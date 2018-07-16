import os
import numpy as np
from os import path
from glob import glob
from sklearn.externals import joblib
from sklearn.tree import export_graphviz
from pandas import read_csv, DataFrame, concat


class TreeAnalysis:

    @staticmethod
    def grow_trees(model, type_path, max_depth=10, min_samples_split=0.1):
        """Create DecisionTrees for each data set.

        :param model: DecisionTreeRegressor or DecisionTreeClassifier
            Model used to create trees.

        :param type_path: Path object
            Path class used to save data.

        :param max_depth: int (default 10)
            Maximum depth of the tree.

        :param min_samples_split: int, float (default 0.1)
            Minimum samples required to split.
            If int, the number of exact samples.
            If float, the percentage of samples to be used for split.

        :return: None
        """
        model.max_depth = max_depth
        model.min_samples_split = min_samples_split

        dataset_paths = glob(path.join(type_path.data_path, '*'))

        for dataset_path in dataset_paths:
            data = read_csv(dataset_path)
            features = data.columns.values[:-1]
            data = data.values

            x = data[:, :-1]
            y = data[:, -1]

            model.fit(x, y)

            dataset_name = dataset_path.split('/')[-1]
            tree_name = dataset_name[:-4]

            # Save as text file
            tree_path = path.join(type_path.text_trees_path, tree_name + '.dot')
            export_graphviz(model,
                            tree_path,
                            feature_names=features,
                            filled=True)

            # Save as png
            png_tree_path = path.join(type_path.visible_trees_path,
                                      tree_name + '.png')

            os.system('dot -Tpng ' + tree_path + ' -o ' + png_tree_path)

            # Save as object
            obj_tree_path = path.join(type_path.object_trees_path,
                                      tree_name + '.pkl')

            joblib.dump(model.tree_, obj_tree_path)

    @staticmethod
    def get_important_nodes(analysis_datapath, type_path):
        """Get the most important nodes.

        :param analysis_datapath: string
            Dataset folder.

        :param type_path: Path object
            Path class used to save data.

        :return: None
        """
        data = read_csv(analysis_datapath, header=0)
        features_names = data.columns.values

        trees_filepath = glob(path.join(type_path.object_trees_path, '*'))

        important_nodes = {}
        columns = ['node_index',
                   'feature',
                   'importance',
                   'impurity',
                   'threshold',
                   'value',
                   'child_left_diff',
                   'child_right_diff']

        for tree_filepath in trees_filepath:
            tree = joblib.load(tree_filepath)
            data = DataFrame(columns=columns)

            importances = tree.compute_feature_importances()
            features = np.argsort(importances)[::-1]

            for feature_i in features:

                if importances[feature_i] == 0:
                    break

                possible_nodes = np.where(tree.feature == feature_i)[0]

                node_index = possible_nodes[0]  # first node with this feature
                left_i = tree.children_left[node_index]    # his left child
                right_i = tree.children_right[node_index]  # his right child

                feature = features_names[feature_i]
                importance = importances[feature_i]

                value = tree.value[node_index][0, 0]
                impurity = tree.impurity[node_index]
                threshold = tree.threshold[node_index]

                child_left_diff = tree.value[left_i][0, 0] - value
                child_right_diff = tree.value[right_i][0, 0] - value

                ins = DataFrame([node_index,
                                 feature,
                                 importance,
                                 impurity,
                                 threshold,
                                 value,
                                 child_left_diff,
                                 child_right_diff], index=columns).T

                data = data.append(ins, ignore_index=True)

            tree_name = tree_filepath.split('/')[-1][:-4]
            important_nodes[tree_name] = data

        concat(important_nodes).to_csv(path.join(type_path.trees_path,
                                                 'important_nodes.csv'))
