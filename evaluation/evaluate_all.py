import warnings

from main import main
from glob import glob
from time import time

warnings.filterwarnings("ignore")

for dataset in ['plates', 'theorem']:  # glob('datasets/*'):
    dataset = 'datasets/' + dataset + '_last.csv'
    dataset_name = dataset.replace('datasets/', '')[:-4]

    overlap = float(0)
    for overlap in range(1):
        t = int(time())

        print('{}% {}'.format(overlap*10, dataset))

        main({'dataset_path': dataset, 'params_path': None, 'overlap': overlap / 10})

        end = (int(time()) - t) // 60
        print('Completed in {} minutes.'.format(end))

        overlap += 0.1
