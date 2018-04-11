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


def main(args):
    dataset_name = get_dataset_name(args['dataset_path'])
    class_column = get_class_column_by_name(dataset_name)

    if args['params_path'] is None:
        obs = Observer(args['dataset_path'], class_column)

        if obs.n_targets() == 2:  # binary
            params_path = 'tests/binary.json'
        else:  # multiclass
            params_path = 'tests/multiclass.json'
    else:
        params_path = args['params_path']

    # Create test folder if not exists
    i = 0
    result_path = 'tests/{}_{}'.format(dataset_name[:-4], i)
    while os.path.exists(result_path):
        i += 1
        result_path = result_path[:-1] + str(i)

    os.makedirs(result_path)

    # Load params and run test
    params = open(params_path, 'r')
    p = json.load(params)
    params.close()

    p['dataset'] = args['dataset_path']
    p['class_column'] = class_column
    p['result_path'] = result_path

    if args['overlap'] is not None:
        p['overlap'] = float(args['overlap'])

    # Save params
    file = open('{}/params.json'.format(result_path), 'w')
    json.dump(p, file)
    file.close()

    run_test(p)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        dest="dataset_path",
                        help="Dataset's absolute/relative path.",
                        required=True)

    parser.add_argument("-p", "--params",
                        default=None,
                        dest="params_path",
                        help=".json params file's absolute/relative path.")

    parser.add_argument("-o", "--overlap",
                        default=None,
                        dest="overlap",
                        help="\% of overlaped features, value between 0.0 and 1.0.")

    # Validate params
    args = vars(parser.parse_args())

    main(args)
