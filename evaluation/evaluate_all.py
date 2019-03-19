import warnings

from main import main
from glob import glob


warnings.filterwarnings("ignore")

tested = [folder.replace('tests/', '') for folder in glob('tests/*')]
tested = ','.join(tested)

for dataset in glob('datasets/*'):
    dataset_name = dataset.replace('datasets/', '')[:-4]

    overlap = float(0)
    while overlap < 1:
        dt_file = dataset_name + '_' + str(int(overlap * 10))
        if dt_file not in tested:
            print('{} {}'.format(overlap, dataset))
            main({'dataset_path': dataset, 'params_path': None, 'overlap': overlap})

        overlap += 0.1
