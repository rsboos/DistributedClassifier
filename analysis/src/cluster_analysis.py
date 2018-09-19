from pandas import read_csv, DataFrame, concat
from .path import RegressionPath
from os import path

class ClusterAnalysis:

    @staticmethod
    def clusters():
        return {
            'cluster1': ['yeast', 'vowel', 'g1318-95-9', 'cancer', 'sharkattack',
                         'iris', 'soybean', 'segmentation', 'glass'],
            'cluster2': ['g2827-95-4', 'optdigits', 'sat', 'crack', 'heroine', 'amyl',
                         'katemine', 'vsa', 'meth', 'caff', 'seed', 'unbalanced1140-68-6',
                         'lsd', 'legalh', 'amphet', 'benzos', 'mushrooms', 'coke', 'ecstasy',
                         'alcohol', 'cannabis', 'nicotine'],
            'cluster3': ['unbalanced9636-40-6', 'diabetics', 'brokenmachine', 'unbalanced8725-53-8'],
            'cluster4': ['unbalanced5627-61-6', 'pageblocks', 'unbalanced5394-89-5'],
            'cluster5': ['activityrecog', 'motionsense', 'pendigits', 'letter'],
            'cluster6': ['banking', 'g8848-86-2', 'g5946-47-2', 'g6576-46-2']
        }

    @classmethod
    def dataset_cluster(cls):
        clusters = cls.clusters()
        datasets = {}

        for c in clusters:
            for dt in clusters[c]:
                datasets[dt] = c

        return datasets

    def process(self):
        data = read_csv('data/datasets.csv', header=0, index_col=None)

        indexes = data.iloc[:, -1].values
        data = data.drop('dataset', axis=1)
        data.index = indexes

        result_mean = DataFrame(columns=data.columns.values)
        result_std = DataFrame(columns=data.columns.values)

        result_max = DataFrame(columns=data.columns.values)
        result_min = DataFrame(columns=data.columns.values)
        for name, cluster in self.clusters().items():
            mean = DataFrame(data.loc[cluster, :].mean(), columns=[name]).T
            std = DataFrame(data.loc[cluster, :].std(), columns=[name]).T

            max = DataFrame(data.loc[cluster, :].max(), columns=[name]).T
            min = DataFrame(data.loc[cluster, :].min(), columns=[name]).T

            result_mean = result_mean.append(mean)
            result_std = result_std.append(std)

            result_max = result_max.append(max)
            result_min = result_min.append(min)

        # Mean and std
        dpath = RegressionPath()
        df = concat([result_mean, result_std], keys=['mean', 'std']).T
        df.to_csv(path.join(dpath.cluster_analysis, 'cluster_features_mean_std.csv'), header=True, index=[0,1])

        # Max and min
        df = concat([result_min, result_max], keys=['min', 'max']).T
        df.to_csv(path.join(dpath.cluster_analysis, 'cluster_features_min_max.csv'), header=True, index=[0, 1])

        # Intervals from mean and std
        lower = result_mean - result_std
        higher = result_mean + result_std

        df = concat([lower, higher], keys=['lower', 'higher']).T
        df.to_csv(path.join(dpath.cluster_analysis, 'cluster_features_intervals.csv'), header=True, index=[0, 1])