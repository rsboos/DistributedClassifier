import os
from glob import glob


for dataset in glob('datasets/*'):

    overlap = float(0)
    while overlap < 1:
        os.system('python3 main.py --dataset {} --overlap {}'.format(dataset, overlap))
        overlap += 0.1
