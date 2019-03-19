import warnings

from main import main
from glob import glob
from time import time

warnings.filterwarnings("ignore")

for dataset in glob('datasets/*'):
    dataset_name = dataset.replace('datasets/', '')[:-4]

    overlap = float(0)
    while overlap < 1:
        tested = [folder.replace('tests/', '') for folder in glob('tests/*')]

        dt_file = dataset_name + '_' + str(int(overlap * 10))

        if dt_file not in tested:
            t = int(time())

            print('{} {}'.format(overlap, dataset))

            main({'dataset_path': dataset, 'params_path': None, 'overlap': overlap})

            end = (int(time()) - t) // 60
            print('Completed in {} minutes.'.format(end))

        overlap += 0.1
