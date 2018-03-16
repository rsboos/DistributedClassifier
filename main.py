import os
import json
import argparse
import numpy as np

from theobserver import Observer
from pandas import DataFrame, concat
from src.agents import Voter, Combiner, Arbiter, Mathematician
from src.selectors import MetaDiff, MetaDiffInc, MetaDiffIncCorr
from src.test import test, load_imports, split_parts, load_scorers


def get_class_column_by_name(name):
    parts = name.split('_')
    parts = parts[-1].split('.')

    if parts[0] == 'first':
        return 0
    elif parts[0] == 'last':
        return -1
    else:
        raise ValueError('Could not defined class column by dataset\'s name: {}.'.format(name))


def get_dataset_name(path):
    if path.endswith('/'):
        path = path[:-1]

    parts = path.split('/')
    return parts[-1]


def run_test(p):
    # Evaluate classifiers
    classifiers = load_imports(p['classifiers'])

    # Evaluate metrics
    scorers = load_scorers(p['metrics'])

    # Evaluate aggregator
    voter = Voter(list(p['voter'].values()))
    combiner = Combiner(load_imports(p['combiner']))
    mathematician = Mathematician(p['mathematician'])
    selectors = [eval(s) for s in p['selection_rules'].values()]
    arbiter = Arbiter(selectors, load_imports(p['arbiter']))

    # Get names
    mathematician_names = list()
    for names in p['mathematician'].values():
        mathematician_names += [name for name in names]

    arbiter_names = []
    arb_methods = list(p['arbiter'].keys())
    for arb in arb_methods:
        for name in p['selection_rules'].keys():
            arbiter_names.append(arb + '_' + name)

    classif_names = list(p['classifiers'].keys())
    combiner_names = list(p['combiner'].keys())
    voter_names = list(p['voter'].keys())

    names = classif_names + voter_names + combiner_names + arbiter_names + mathematician_names

    # Run test
    test(overlap=p['overlap'],
         filepath=p['dataset'],
         iterations=p['iterations'],
         class_column=p['class_column'],
         random_state=p['random_state'],
         scorers=scorers,
         classifiers=classifiers,
         voter=voter,
         arbiter=arbiter,
         combiner=combiner,
         selectors=selectors,
         mathematician=mathematician,
         names=names,
         results_path=p['result_path'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        dest="dataset_path",
                        help="Dataset's absolute/relative path.",
                        required=True)

    parser.add_argument("-p", "--params",
                        default=None,
                        dest="params_folder",
                        help="Folder where a file params.json is.")

    # Validate params
    args = parser.parse_args()

    if args.params_folder:
        params_path = '{}/params.json'.format(args.params_folder)
        result_path = args.params_folder
    else:
        dataset_name = get_dataset_name(args.dataset_path)
        class_column = get_class_column_by_name(dataset_name)
        obs = Observer(args.dataset_path, class_column)

        if obs.n_targets() == 2:  # binary
            params_path = 'tests/binary.json'
        else:  # multiclass
            params_path = 'tests/multiclass.json'

        # Create test folder if not exists
        i = 0
        result_path = 'tests/{}_{}'.format(dataset_name[:-4], i)
        while os.path.exists(result_path):
            i += 1

        result_path = result_path[:-1] + '_' + str(i)
        os.makedirs(result_path)

        # Copy params file to test folder
        os.system('cp {} {}/params.json'.format(params_path, result_path))

    # Load params and run test
    params = open(params_path, 'r')
    p = json.load(params)
    params.close()

    p['dataset'] = args.dataset_path
    p['result_path'] = result_path

    run_test(p)
