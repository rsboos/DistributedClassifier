import ete3
import numpy as np
import matplotlib.pyplot as plt

from os import path
from glob import glob
from math import fsum
from plotnine import *
from .path import RegressionPath
from .cluster_analysis import ClusterAnalysis
from plotnine.guides import guides, guide_legend
from sklearn.cluster import AgglomerativeClustering
from pandas import read_csv, Series, DataFrame, concat


class Graphics:

    def __init__(self, metric, type_path):
        self.metric = [metric, 'f1'] if 'f1_' in metric else [metric, metric]
        self.tests_path = '../evaluation/research_tests/'
        self.type_path = type_path

    def get_folders(self, clusters, overlap):
        folders = []
        for c in list(clusters):
            found = glob(path.join(self.tests_path, '*' + c + '*' + '_{}'.format(overlap)))
            folders.append(found)

        return list(np.ravel(folders))

    def _get_dataset_performance(self, datasets_folders=[]):
        methods_data = {}
        dataset_clusters = ClusterAnalysis.dataset_cluster()

        if len(datasets_folders) == 0:
            datasets_folders = [p for p in glob(path.join(self.tests_path, '*')) if path.isdir(p)]

        for folder in datasets_folders:
            files = glob(path.join(folder, '*'))
            files = set(files) - {path.join(folder, self.type_path.default_file), path.join(folder, 'params.json')}

            for file in files:
                data = read_csv(file, header=0, index_col=0)

                try:
                    data = data.loc[:, self.metric[0]]
                except KeyError:
                    data = data.loc[:, self.metric[1]]

                ranking = list(data.values)

                path_parts = file.split('/')

                dataset = path_parts[-2]
                dataset = dataset.split('_')[0]
                dataset = dataset_clusters[dataset] + '_' + dataset

                method = self.type_path.fix_method_name(file)
                type_m = self.type_path.concat_method_type(method)

                methods_data.setdefault(type_m, {})
                methods_data[type_m][dataset] = ranking

        return methods_data


class Boxplot(Graphics):

    def __init__(self, metric='f1_macro', type_path=RegressionPath()):
        super().__init__(metric, type_path)
        self.__classifiers = ['gnb', 'dtree', 'svc', 'knn', 'mlp']

    def show(self):
        """Show boxplot."""
        plt.show()

    def save(self, filename):
        """Save figure in Path.graphics_path.

        :param filename: String
            File name.

        :return: None
        """
        plt.savefig(path.join(self.type_path.graphics_path, filename), bbox_inches='tight')

    def ranking(self, cluster='*', overlap='*'):
        """Create a boxplot by ranking."""
        folders = self.get_folders(cluster, overlap)
        r, m = self.__get_ranking(folders)
        self.__make(r, m, 'Rank Position')

    def type_ranking(self, cluster='*', overlap='*'):
        """Create a boxplot with method's types ranking."""
        folders = self.get_folders(cluster, overlap)
        r, m = self.__get_type_ranking(folders)
        self.__make(r, m, 'Rank Position')

    def performance(self, cluster='*', overlap='*'):
        """Create a boxplot by performance."""
        folders = self.get_folders(cluster, overlap)
        r, m = self.__get_performance(folders)
        self.__make(r, m, 'F1 Score')

    def baselined_performance(self, cluster='*', overlap='*'):
        """Create a boxplot by performance."""
        folders_overlap = self.get_folders(cluster, overlap)
        folders_baseline = self.get_folders(cluster, 10)

        rank_overlap, method_overlap = self.__get_performance(folders_overlap)
        rank_baseline, method_baseline = self.__get_performance(folders_baseline)

        ranking = []
        for i in range(len(rank_baseline)):
            base = np.array(rank_baseline[i])
            overlap = np.array(rank_overlap[i])

            if len(base) != len(overlap):
                del method_baseline[i]
                continue

            ranking.append(base - overlap)

        self.__make(ranking, method_baseline, 'F1 Score', ticks=[i/10 for i in range(-10, 11)])

    def type_performance(self, cluster='*', overlap='*'):
        """Create a boxplot with method's types performance."""
        folders = self.get_folders(cluster, overlap)
        r, m = self.__get_type_performance(folders)
        self.__make(r, m, 'F1 Score')

    def baselined_type_performance(self, cluster='*', overlap='*'):
        """Create a boxplot with method's types performance."""
        folders_overlap = self.get_folders(cluster, overlap)
        folders_baseline = self.get_folders(cluster, 10)

        rank_overlap, method_overlap = self.__get_type_performance(folders_overlap)
        rank_baseline, method_baseline = self.__get_type_performance(folders_baseline)

        ranking = []
        for i in range(len(rank_baseline)):
            base = np.array(rank_baseline[i])
            overlap = np.array(rank_overlap[i])

            if len(base) != len(overlap):
                del method_baseline[i]
                continue

            ranking.append(base - overlap)

        self.__make(ranking, method_baseline, 'F1 Score', ticks=[i/10 for i in range(-10, 11)])

    def dataset_performance(self, overlap='*'):
        """Create a boxplot with method's performance according to datasets."""
        folders = self.get_folders('*', overlap)
        methods_data = self._get_dataset_performance(folders)

        for method, datasets in methods_data.items():
            positions = list(datasets.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods, ranking = zip(*positions)

            self.__make(ranking, methods, 'F1 Score', 'Datasets')

            named_overlap = '-' + str(overlap) if overlap != '*' else ''
            self.save('bp-performance-{}{}.pdf'.format(method, named_overlap))
            plt.close()

    def baselined_dataset_performance(self, overlap='*'):
        """Create a boxplot with method's performance according to datasets."""
        folders_overlap = self.get_folders('*', overlap)
        folders_baseline = self.get_folders('*', 10)

        methods_data_overlap = self._get_dataset_performance(folders_overlap)
        methods_data_baseline = self._get_dataset_performance(folders_baseline)

        for method in methods_data_baseline:

            datasets_overlap = methods_data_overlap[method]
            positions_overlap = list(datasets_overlap.items())
            positions_overlap = sorted(positions_overlap, key=lambda x: x[0])
            methods_overlap, rank_overlap = zip(*positions_overlap)

            datasets_baseline = methods_data_baseline[method]
            positions_baseline = list(datasets_baseline.items())
            positions_baseline = sorted(positions_baseline, key=lambda x: x[0])
            methods_baseline, rank_baseline = zip(*positions_baseline)
            methods_baseline = list(methods_baseline)

            ranking = []
            for i in range(len(rank_overlap)):
                base = np.array(rank_baseline[i])
                overlaps = np.array(rank_overlap[i])

                if len(base) != len(overlaps):
                    del methods_baseline[i]
                    continue

                ranking.append(base - overlaps)

            self.__make(ranking, methods_baseline, 'F1 Score', 'Datasets', ticks=[i/10 for i in range(-10, 11)])

            named_overlap = '-' + str(overlap) if overlap != '*' else ''
            self.save('bp-baseline-performance-{}{}.pdf'.format(method, named_overlap))
            plt.close()

    def cluster_performance(self, overlap='*'):
        """Create a boxplot with method's performance according to cluster's datasets."""
        folders = self.get_folders('*', overlap)
        clusters_data = self.__get_cluster_performance(folders)

        for method, clusters in clusters_data.items():
            positions = list(clusters.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods, ranking = zip(*positions)

            self.__make(ranking, methods, 'F1 Score', 'Clusters')

            named_overlap = '-' + str(overlap) if overlap != '*' else ''
            self.save('bp-performance-clusters-{}{}.pdf'.format(method, named_overlap))
            plt.close()

    def baselined_cluster_performance(self, overlap='*'):
        """Create a boxplot with method's performance according to cluster's datasets."""
        folders_overlap = self.get_folders('*', overlap)
        folders_baseline = self.get_folders('*', 10)

        clusters_data_overlap = self.__get_cluster_performance(folders_overlap)
        clusters_data_baseline = self.__get_cluster_performance(folders_baseline)

        for method in clusters_data_overlap:

            clusters_overlap = clusters_data_overlap[method]
            positions = list(clusters_overlap.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods_overlap, rank_overlap = zip(*positions)

            clusters_baseline = clusters_data_baseline[method]
            positions = list(clusters_baseline.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods_baseline, rank_baseline = zip(*positions)
            methods_baseline = list(methods_baseline)

            ranking = []
            for i in range(len(rank_overlap)):
                base = np.array(rank_baseline[i])
                overlaps = np.array(rank_overlap[i])

                if len(base) != len(overlaps):
                    del methods_baseline[i]
                    continue

                ranking.append(base - overlaps)

            self.__make(ranking, methods_baseline, 'F1 Score', 'Clusters', ticks=[i/10 for i in range(-10, 11)])

            named_overlap = '-' + str(overlap) if overlap != '*' else ''
            self.save('bp-baseline-performance-clusters-{}{}.pdf'.format(method, named_overlap))
            plt.close()

    def dataset_method_performance(self, overlap='*'):
        """Create a boxplot with methods' performances for each dataset."""
        folders = self.get_folders('*', overlap)
        datasets_data = self.__get_dataset_method_performance(folders)
        clusters = ClusterAnalysis.dataset_cluster()

        for dataset, methods in datasets_data.items():
            positions = list(methods.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods, ranking = zip(*positions)

            self.__make(ranking, methods, 'F1 Score', 'Methods')

            cluster = clusters[dataset.split('_')[0]]
            self.save('bp-performance-dt-{}-{}.pdf'.format(cluster, dataset))
            plt.close()

    def baselined_dataset_method_performance(self, overlap='*'):
        """Create a boxplot with methods' performances for each dataset."""
        folders_overlap = self.get_folders('*', overlap)
        folders_baseline = self.get_folders('*', 10)

        datasets_data_overlap = self.__get_dataset_method_performance(folders_overlap)
        datasets_data_baseline = self.__get_dataset_method_performance(folders_baseline)

        clusters = ClusterAnalysis.dataset_cluster()

        for dataset in datasets_data_overlap:

            methods = datasets_data_overlap[dataset]
            positions = list(methods.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods_overlap, rank_overlap = zip(*positions)

            dt = dataset.split('_')
            dt[-1] = '10'
            dt = '_'.join(dt)

            methods = datasets_data_baseline[dt]
            positions = list(methods.items())
            positions = sorted(positions, key=lambda x: x[0])
            methods_baseline, rank_baseline = zip(*positions)

            ranking = []
            for i in range(len(rank_overlap)):
                base = np.array(rank_baseline[i])
                overlaps = np.array(rank_overlap[i])

                if len(base) != len(overlaps):
                    del methods_baseline[i]
                    continue

                ranking.append(base - overlaps)

            self.__make(ranking, methods_baseline, 'F1 Score', 'Methods', ticks=[i/10 for i in range(-10, 11)])

            cluster = clusters[dataset.split('_')[0]]
            self.save('bp-baseline-performance-dt-{}-{}.pdf'.format(cluster, dataset))
            plt.close()

    def __get_ranking(self, datasets_folders=[]):
        rankings = []

        if len(datasets_folders) == 0:
            datasets_folders = [p for p in glob(path.join(self.tests_path, '*')) if path.isdir(p)]

        for folder in datasets_folders:
            data = read_csv(path.join(folder, self.type_path.default_file),
                            header=[0, 1], index_col=0)

            try:
                data = data.loc[:, 'mean'].loc[:, self.metric[0]]
            except KeyError:
                data = data.loc[:, 'mean'].loc[:, self.metric[1]]

            data = data.sort_values()
            methods = list(data.index.values)

            i = 0
            while len(methods) < 32:
                methods.append(self.__classifiers[i])
                i += 1

            methods = list(map(lambda x: self.type_path.concat_method_type(x), methods))
            positions = list(enumerate(methods))

            # Sort by name
            positions.sort(key=lambda x: x[1])
            ranking, ordered_methods = zip(*positions)

            # From 1 to 32
            ranking = list(map(lambda x: x + 1, ranking))
            rankings.append(ranking)

        return rankings, list(ordered_methods)

    def __get_type_ranking(self, datasets_folders=[]):
        rankings = []

        if len(datasets_folders) == 0:
            datasets_folders = [p for p in glob(path.join(self.tests_path, '*')) if path.isdir(p)]

        for folder in datasets_folders:
            data = read_csv(path.join(folder, self.type_path.default_file),
                            header=[0, 1], index_col=0)

            try:
                data = data.loc[:, 'mean'].loc[:, self.metric[0]]
            except KeyError:
                data = data.loc[:, 'mean'].loc[:, self.metric[1]]

            methods = list(map(lambda x: self.type_path.concat_method_type(x), data.index.values))
            data.index = methods

            method_types = {}
            for m in methods:
                method_type = m.split('_')[0]
                method_types.setdefault(method_type, [])
                method_types[method_type].append(data.loc[m])

            for k in method_types:
                method_types[k] = np.mean(method_types[k])

            data = Series(method_types)
            data = data.sort_values()
            positions = list(enumerate(data.index.values))

            # Sort by name
            positions.sort(key=lambda x: x[1])
            ranking, ordered_methods = zip(*positions)

            # From 1 to 32
            ranking = list(map(lambda x: x + 1, ranking))
            rankings.append(ranking)

        return rankings, list(ordered_methods)

    def __get_performance(self, datasets_folders=[]):
        methods_data = {}

        if len(datasets_folders) == 0:
            datasets_folders = [p for p in glob(path.join(self.tests_path, '*')) if path.isdir(p)]

        for folder in datasets_folders:
            files = glob(path.join(folder, '*'))
            files = set(files) - {path.join(folder, self.type_path.default_file), path.join(folder, 'params.json')}

            for file in files:
                data = read_csv(file, header=0, index_col=0)

                try:
                    data = data.loc[:, self.metric[0]]
                except KeyError:
                    data = data.loc[:, self.metric[1]]

                ranking = list(data.values)

                method = self.type_path.fix_method_name(file)
                type_m = self.type_path.concat_method_type(method)

                methods_data.setdefault(type_m, [])
                methods_data[type_m] += ranking

        positions = list(methods_data.items())
        positions = sorted(positions, key=lambda x: x[0])
        methods, ranking = zip(*positions)

        return list(ranking), list(methods)

    def __get_type_performance(self, datasets_folders=[]):
        ranking, ordered_methods = self.__get_performance(datasets_folders)

        method_types = {}
        for method, ranking in zip(ordered_methods, ranking):
            type_m = method.split('_')[0]
            method_types.setdefault(type_m, [])
            method_types[type_m] += ranking

        positions = list(method_types.items())
        positions = sorted(positions, key=lambda x: x[0])
        methods, ranking = zip(*positions)

        return list(ranking), list(methods)

    def __get_cluster_performance(self, datasets_folders=[]):
        methods_data = self._get_dataset_performance(datasets_folders)
        clusters_data = {}

        for method, data in methods_data.items():
            for dataset in data.keys():
                cluster = dataset.split('_')[0]
                clusters_data.setdefault(method, {})
                clusters_data[method].setdefault(cluster, [])
                clusters_data[method][cluster] += data[dataset]

        return clusters_data

    def __get_dataset_method_performance(self, datasets_folders=[]):
        datasets_data = {}

        if len(datasets_folders) == 0:
            datasets_folders = [p for p in glob(path.join(self.tests_path, '*')) if path.isdir(p)]

        for folder in datasets_folders:
            files = glob(path.join(folder, '*'))
            files = set(files) - {path.join(folder, self.type_path.default_file), path.join(folder, 'params.json')}

            last_folder = folder.split('/')[-1]
            datasets_data[last_folder] = {}

            for file in files:
                data = read_csv(file, header=0, index_col=0)

                try:
                    data = data.loc[:, self.metric[0]]
                except KeyError:
                    data = data.loc[:, self.metric[1]]

                ranking = list(data.values)

                method = self.type_path.fix_method_name(file)
                type_m = self.type_path.concat_method_type(method)

                datasets_data[last_folder].setdefault(type_m, [])
                datasets_data[last_folder][type_m] += ranking

        return datasets_data

    def __make(self, boxplot_data, ordered_methods, ylabel, xlabel='Methods', ticks=None):
        data_sum = [fsum(v) / len(v) for v in boxplot_data]
        mean = fsum(data_sum) / len(data_sum)

        if ticks is None:
            ticks = [i / 10 for i in range(0, 11)]
        elif mean <= 1:
            ticks = [i / 10 for i in range(-10, 11)]
        else:
            ticks = range(int(np.min(boxplot_data)), int(np.max(boxplot_data)) + 1)

        ticks = list(ticks)

        fig, ax = plt.subplots()
        ax.boxplot(boxplot_data)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ordered_methods, rotation=90)

        fig.set_size_inches(18.5, 10.5, forward=True)
        plt.yticks(ticks)


class NewickTree(Graphics):
    """
    =============================================================================
    Various Agglomerative Clustering on a 2D embedding of digits
    =============================================================================

    An illustration of various linkage option for agglomerative clustering on
    a 2D embedding of the digits dataset.

    The goal of this example is to show intuitively how the metrics behave, and
    not to find good clusters for the digits. This is why the example works on a
    2D embedding.

    What this example shows us is the behavior "rich getting richer" of
    agglomerative clustering that tends to create uneven cluster sizes.
    This behavior is especially pronounced for the average linkage strategy,
    that ends up with a couple of singleton clusters.

    # Authors: Gael Varoquaux
    # License: BSD 3 clause (C) INRIA 2014
    """
    def create(self, linkage='ward'):
        """
        Create a Newick Tree and display it.

        :param linkage: str {(Default 'ward'), 'average', 'complete'}
            Agglomerative method.

        :return: None
        """
        data = read_csv('data/datasets.csv')
        X = data.values[:, :-1]
        y = data.values[:, -1]

        np.random.seed(0)

        clusterer = AgglomerativeClustering(linkage=linkage, n_clusters=X.shape[0])
        clusterer.fit(X)

        spanner = self.__get_cluster_spanner(clusterer)
        newick_tree = self.__build_newick_tree(clusterer.children_, clusterer.n_leaves_, X, y, spanner)
        tree = ete3.Tree(newick_tree)
        tree.show()

    def __build_newick_tree(self, children, n_leaves, X, leaf_labels, spanner):
        """
        build_Newick_tree(children,n_leaves,X,leaf_labels,spanner)

        Get a string representation (Newick tree) from the sklearn
        AgglomerativeClustering.fit output.

        Input:
            children: AgglomerativeClustering.children_
            n_leaves: AgglomerativeClustering.n_leaves_
            X: parameters supplied to AgglomerativeClustering.fit
            leaf_labels: The label of each parameter array in X
            spanner: Callable that computes the dendrite's span

        Output:
            ntree: A str with the Newick tree representation

        """
        return self.__go_down_tree(children, n_leaves, X, leaf_labels, len(children) + n_leaves - 1, spanner)[0] + ';'

    def __go_down_tree(self, children, n_leaves, X, leaf_labels, nodename, spanner):
        """
        go_down_tree(children,n_leaves,X,leaf_labels,nodename,spanner)

        Iterative function that traverses the subtree that descends from
        nodename and returns the Newick representation of the subtree.

        Input:
            children: AgglomerativeClustering.children_
            n_leaves: AgglomerativeClustering.n_leaves_
            X: parameters supplied to AgglomerativeClustering.fit
            leaf_labels: The label of each parameter array in X
            nodename: An int that is the intermediate node name whos
                children are located in children[nodename-n_leaves].
            spanner: Callable that computes the dendrite's span

        Output:
            ntree: A str with the Newick tree representation

        """
        nodeindex = nodename - n_leaves
        if nodename < n_leaves:
            return leaf_labels[nodeindex], np.array([X[nodeindex]])
        else:
            node_children = children[nodeindex]
            branch0, branch0samples = self.__go_down_tree(children, n_leaves, X, leaf_labels, node_children[0], spanner)
            branch1, branch1samples = self.__go_down_tree(children, n_leaves, X, leaf_labels, node_children[1], spanner)
            node = np.vstack((branch0samples, branch1samples))
            branch0span = spanner(branch0samples)
            branch1span = spanner(branch1samples)
            nodespan = spanner(node)
            branch0distance = nodespan - branch0span
            branch1distance = nodespan - branch1span
            nodename = '({branch0}:{branch0distance},{branch1}:{branch1distance})'.format(branch0=branch0,
                                                                                          branch0distance=branch0distance,
                                                                                          branch1=branch1,
                                                                                          branch1distance=branch1distance)
            return nodename, node

    def __get_cluster_spanner(self, aggClusterer):
        """
        spanner = get_cluster_spanner(aggClusterer)

        Input:
            aggClusterer: sklearn.cluster.AgglomerativeClustering instance

        Get a callable that computes a given cluster's span. To compute
        a cluster's span, call spanner(cluster)

        The cluster must be a 2D numpy array, where the axis=0 holds
        separate cluster members and the axis=1 holds the different
        variables.

        """
        if aggClusterer.linkage == 'ward':
            if aggClusterer.affinity == 'euclidean':
                spanner = lambda x: np.sum((x - aggClusterer.pooling_func(x, axis=0)) ** 2)
        elif aggClusterer.linkage == 'complete':
            if aggClusterer.affinity == 'euclidean':
                spanner = lambda x: np.max(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
            elif aggClusterer.affinity == 'l1' or aggClusterer.affinity == 'manhattan':
                spanner = lambda x: np.max(np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=2))
            elif aggClusterer.affinity == 'l2':
                spanner = lambda x: np.max(np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)))
            elif aggClusterer.affinity == 'cosine':
                spanner = lambda x: np.max(np.sum((x[:, None, :] * x[None, :, :])) / (
                            np.sqrt(np.sum(x[:, None, :] * x[:, None, :], axis=2, keepdims=True)) * np.sqrt(
                        np.sum(x[None, :, :] * x[None, :, :], axis=2, keepdims=True))))
            else:
                raise AttributeError('Unknown affinity attribute value {0}.'.format(aggClusterer.affinity))
        elif aggClusterer.linkage == 'average':
            if aggClusterer.affinity == 'euclidean':
                spanner = lambda x: np.mean(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
            elif aggClusterer.affinity == 'l1' or aggClusterer.affinity == 'manhattan':
                spanner = lambda x: np.mean(np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=2))
            elif aggClusterer.affinity == 'l2':
                spanner = lambda x: np.mean(np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)))
            elif aggClusterer.affinity == 'cosine':
                spanner = lambda x: np.mean(np.sum((x[:, None, :] * x[None, :, :])) / (
                            np.sqrt(np.sum(x[:, None, :] * x[:, None, :], axis=2, keepdims=True)) * np.sqrt(
                        np.sum(x[None, :, :] * x[None, :, :], axis=2, keepdims=True))))
            else:
                raise AttributeError('Unknown affinity attribute value {0}.'.format(aggClusterer.affinity))
        else:
            raise AttributeError('Unknown linkage attribute value {0}.'.format(aggClusterer.linkage))

        return spanner


class GGPlot(Graphics):

    def __init__(self, metric='f1_macro', type_path=RegressionPath()):
        super().__init__(metric, type_path)

    def dataset_by_methods(self, overlap='*'):
        """Create a GGPlot with method's performance according to datasets."""
        data = {}
        all_data = DataFrame(columns=['method', 'dataset', 'median', 'std'])
        folders = self.get_folders('*', overlap)
        methods_data = self._get_dataset_performance(folders)

        for method in methods_data:
            method_parts = method.split('_')
            mtype = method_parts[0]
            method_name = '_'.join(method_parts[1:])

            data.setdefault(mtype, DataFrame(columns=['method', 'dataset', 'median', 'std']))

            for dataset in methods_data[method]:
                values = methods_data[method][dataset]

                median = float(np.median(values))
                std = float(np.std(values))

                ins = {'method': method_name,
                       'dataset': dataset,
                       'median': median,
                       'std': std}

                data[mtype] = data[mtype].append(ins, ignore_index=True)

                ins['method'] = method
                all_data = all_data.append(ins, ignore_index=True)

        data['all'] = all_data
        for mtype in data:
            data[mtype] = data[mtype].sort_values('dataset')

            if mtype == 'all':
                data[mtype] = data[mtype].sort_values('method')

            g = ggplot(aes(x='dataset', y='method', size='std', color='median'), data=data[mtype]) + \
                geom_point() + \
                scale_color_gradient(low='#BDDCFA', high='#00376D') + \
                guides(color=guide_legend()) + \
                xlab("Dataset") + \
                ylab("Method") + \
                theme_minimal() + \
                scale_size(rescaler=lambda x, _from: pow(1 - (x - _from[0]) / (_from[1] - _from[0]), 3)) + \
                theme(axis_text_x=element_text(angle=90))

            named_overlap = '-' + str(overlap) if overlap != '*' else ''
            g.save(filename=path.join(self.type_path.graphics_path, 'method-dataset-{}{}.pdf'.format(mtype, named_overlap)),
                   width=16.5,
                   height=10.5)


class Histogram(Graphics):

    def __init__(self, metric='f1_macro', type_path=RegressionPath()):
        super().__init__(metric, type_path)

    def feature_by_cluster(self):
        """Create a Histrogram of datasets characteristics divided by cluster."""
        datasets_char = read_csv('data/datasets_clusters.csv', header=0)

        # dt_clusters = ClusterAnalysis.dataset_cluster()
        # clusters_set = set(dt_clusters.values())

        for feature in datasets_char.columns.values:

            # for cluster in clusters_set:
            #     data = datasets_char.where(datasets_char.loc[:, 'cluster'] == cluster)
            #     data = data.dropna()

            g = ggplot(datasets_char, aes(feature, y='..scaled..', fill='cluster', color='cluster')) + \
                geom_density(alpha=0.1) + \
                xlab(feature.replace('_', ' ').capitalize()) + \
                ylab("Density") + \
                theme_minimal()

            g.save(filename=path.join(self.type_path.graphics_path, 'hist-density-{}.png'.format(feature)),
                   width=16.5,
                   height=10.5)
