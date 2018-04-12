import warnings

from main import main
from glob import glob


warnings.filterwarnings("ignore")

for dataset in glob('datasets/*'):

    overlap = float(0)
    while overlap < 1:
        print('{} {}'.format(overlap, dataset))
        main({'dataset_path': dataset, 'params_path': None, 'overlap': overlap})
        overlap += 0.1
