from pandas import DataFrame, read_csv, concat
from theobserver import Observer
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sbn
import numpy as np
import os


class PartitionAnalysis:

    @staticmethod
    def characteristics():
        """Group the partitions characteristics."""
        chars = {}
        partition_files = glob('../evaluation/datasets_partitions/*.csv')

        for filepath in partition_files:
            dt_filename = filepath.split('/')[-1]
            dt = read_csv(os.path.join('../evaluation/datasets', dt_filename), header=None, index_col=None)

            dt_name = dt_filename[:-4]
            chars[dt_name] = []
            with open(filepath, 'r') as file:

                for line in file:
                    p = np.array([int(i.split('.')[0]) for i in line.split(',') if len(i) > 0 and i != '\n'])

                    p_dt = concat([dt.iloc[:, p], dt.iloc[:, -1]], ignore_index=True, axis=1)
                    p_dt.to_csv('temp.csv', header=False, index=False)

                    obs = Observer('temp.csv')
                    extracted_chars = obs.extract()[:-2]
                    chars[dt_name].append(extracted_chars)

        chars_summary = []
        for dt_name in chars:
            grouped_chars = np.array(chars[dt_name])

            dt_chars = []
            dt_chars.append(np.max(grouped_chars[:, 0]))   # instances
            dt_chars.append(np.sum(grouped_chars[:, 1]) / 10)   # features
            dt_chars.append(np.max(grouped_chars[:, 2]))   # targets
            dt_chars.append(np.mean(grouped_chars[:, 3]))  # silhouette
            dt_chars.append(np.mean(grouped_chars[:, 4]))  # imbalance
            dt_chars.append(np.mean(grouped_chars[:, 5]) / 10)  # binary features
            dt_chars.append(np.max(grouped_chars[:, 6]))   # majority
            dt_chars.append(np.max(grouped_chars[:, 7]))   # minority
            dt_chars.append(dt_name)                       # dataset

            chars_summary.append(dt_chars)

        with open('data/datasets.csv', 'r') as file:
            header = file.readline()[:-1].split(',')

        DataFrame(chars_summary, columns=header).to_csv('data/datasets_real.csv', header=True, index=False)
        os.system('rm -f temp.csv')

    @staticmethod
    def compare():
        """Compare partitions' characteristics to centralized datasets' characteristics."""
        centralized = read_csv('data/datasets.csv', header=0, index_col=None)
        decentralized = read_csv('data/datasets_real.csv', header=0, index_col=None)

        c = DataFrame([['Centralized'] for _ in range(centralized.shape[0])], columns=['Type'])
        d = DataFrame([['Partitioned'] for _ in range(decentralized.shape[0])], columns=['Type'])

        chars = centralized.columns.values
        for j in range(centralized.shape[1] - 1):

            feat_centralized = concat([centralized.iloc[:, [j, -1]], c], axis=1, sort=True)
            feat_decentralized = concat([decentralized.iloc[:, [j, -1]], d], axis=1, sort=True)

            dt = concat([feat_centralized, feat_decentralized], axis=0, sort=False)

            _ = sbn.scatterplot(x='Dataset', y=chars[j], hue='Type', style='Type', data=dt)

            plt.xticks(rotation=90, fontsize=5)
            plt.savefig('tests/partition/{}.pdf'.format(chars[j]), bbox_inches='tight', dpi=300)
            plt.close('all')
