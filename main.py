import json
import argparse
import numpy as np

from src.data import Data
from src.metrics import summary
from pandas import DataFrame, concat
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from src.simulator import FeatureDistributed
from src.agents import Voter, Combiner, Arbiter, Mathematician


def load_imports(imports):
    objs = list()

    for c in imports.values():                           # for each classfier
        i, modules, obj = split_parts(c)                 # separate in parts

        str_eval = "from {} import {}".format(modules, obj[:i])

        print(str_eval)

        # Execute import
        exec(str_eval)

        # Evaluate a classifier
        objs.append(eval(obj))

    return objs


def split_parts(label):
    parts = label.split('.')                         # separate in parts

    i = ['(' in part for part in parts].index(1)     # filter parts

    modules = '.'.join(parts[:i])                    # get modules
    obj = '.'.join(parts[i:])                        # get obj
    i = obj.find('(')                                # filter obj

    return i, modules, obj


if __name__ == "__main__":
    ###########################################################################
    # COMMAND LINE PARAMS #####################################################
    ###########################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--params",
                        dest="params_folder",
                        help="Folder where a file params.json is.",
                        required=True)

    # Validate params
    args = parser.parse_args()

    ###########################################################################
    # LOAD APP PARAMS #########################################################
    ###########################################################################
    print('Loading params...', end=' ')

    params = open("{}/params.json".format(args.params_folder), 'r')
    p = json.load(params)

    print('OK')

    # Evaluate classifiers
    print('Loading classifiers...')

    classifiers = load_imports(p['classifiers'])

    print('Done.')

    # Evaluate metrics
    print('Loading metrics...')
    scorers = dict()

    for n, c in p['metrics'].items():                    # for each metric (name, method)
        i, modules, metric = split_parts(c)              # separate in parts

        params = metric[i + 1:-1]                        # get score func params (without '()')
        metric = metric[:i]                              # get score func name

        str_eval_import = "from {} import {}".format(modules, metric)
        str_eval_scorer = "make_scorer({}, {})".format(metric, params)

        print(str_eval_import)
        print(str_eval_scorer)

        # Execute import
        exec(str_eval_import)

        # Evaluate a metric
        scorers[n] = eval(str_eval_scorer)

    print('Done.')

    # Evaluate aggregator
    print('Loading aggregators...', end=' ')

    voter = Voter(list(p['voter'].values()))
    combiner = Combiner(load_imports(p['combiner']))
    mathematician = Mathematician(p['mathematician'])

    print('OK')
    ###########################################################################
    # SIMULATE DISTRIBUTION ###################################################
    ###########################################################################
    print('Loading dataset {}...'.format(p['dataset']), end=' ')

    data = Data.load(p['dataset'], p['class_column'])

    print('OK')

    # Create simulator (agents' manager)
    print('Simulating distribution...', end=' ')

    simulator = FeatureDistributed.load(data,
                                        classifiers,
                                        p['overlap'],
                                        voter=voter,
                                        combiner=combiner,
                                        mathematician=mathematician)

    print('OK')

    ###########################################################################
    # CROSS VALIDATION ########################################################
    ###########################################################################
    print('Cross validating...', end=' ')

    ranks, scores = simulator.repeated_cv(scorers, p['random_state'], p['iterations'])

    print('OK')

    ###########################################################################
    # SAVE RESULTS ############################################################
    ###########################################################################
    # Save CV scores
    print('Saving CV scores...', end=' ')

    mathematician_names = list()
    for names in p['mathematician'].values():
        mathematician_names += [name for name in names]

    classif_names = list(p['classifiers'].keys())
    combiner_names = list(p['combiner'].keys())
    voter_names = list(p['voter'].keys())

    names = classif_names + voter_names + combiner_names + mathematician_names
    n = len(names)

    [scores[i].to_csv('{}/cv_scores_{}.csv'.format(args.params_folder, names[i])) for i in range(n)]

    print('OK')

    # Create CV summary
    print('Creating CV summary...', end=' ')

    stats = summary(scores)
    stats.index = names
    stats.to_csv('{}/cv_summary.csv'.format(args.params_folder))

    print('OK')
