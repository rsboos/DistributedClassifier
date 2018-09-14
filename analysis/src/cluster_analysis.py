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

    def process(self):
        data = read_csv('data/datasets.csv', header=0, index_col=None)

        indexes = data.iloc[:, -1].values
        data = data.drop('dataset', axis=1)
        data.index = indexes

        result_mean = DataFrame(columns=data.columns.values)
        result_std = DataFrame(columns=data.columns.values)
        for name, cluster in self.clusters().items():
            mean = data.loc[cluster, :].mean()
            std = data.loc[cluster, :].std()

            result_mean = result_mean.append(mean, ignore_index=True)
            result_std = result_std.append(std, ignore_index=True)

        dpath = RegressionPath()
        df = concat([result_mean, result_std], keys=['mean', 'std']).T
        df.to_csv(path.join(dpath.cluster_analysis, 'cluster_features.csv'), header=True, index=[0,1])