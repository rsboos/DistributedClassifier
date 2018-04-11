from main import main
from glob import glob


for dataset in glob('datasets/*'):

    overlap = float(0)
    while overlap < 1:
        main({'dataset_path': dataset, 'params_path': None, 'overlap': overlap})
        overlap += 0.1
