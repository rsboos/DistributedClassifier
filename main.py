import json
import argparse

from src.data import Data
from src.simulator import FeatureDistributed


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

    print('OK', end='\n\n')

    # Evaluate classifiers
    print('Loading classifiers...')

    classifiers = list()

    for c in p['classifiers'].values():                  # for each classfier
        parts = c.split('.')                             # separate in parts

        i = ['(' in part for part in parts].index(1)     # filter parts

        modules = '.'.join(parts[:i])                    # get modules
        classifier = '.'.join(parts[i:])                 # get classifier

        i = classifier.find('(')                         # filter classifier

        print("from {} import {}".format(modules, classifier[:i]))

        # Execute import
        exec("from {} import {}".format(modules, classifier[:i]))

        # Evaluate a classifier
        classifiers.append(eval(classifier))

    print('Done.', end='\n\n')

    # Evaluate metrics
    print('Loading metrics...')
    scorers = dict()

    for n, c in p['metrics'].items():                    # for each metric (name, method)
        parts = c.split('.')                             # separate in parts

        modules = '.'.join(parts[:-1])                   # get modules
        metric = ''.join(parts[-1])                      # get metric

        print("from {} import {}".format(modules, metric))

        # Execute import
        exec("from {} import {}".format(modules, metric))

        # Evaluate a metric
        scorers[n] = eval(metric)

    print('Done.', end='\n\n')

    ###########################################################################
    # SIMULATE DISTRIBUTION ###################################################
    ###########################################################################
    print('Loading dataset {}...'.format(p['dataset']), end=' ')

    data = Data.load(p['dataset'], p['class_column'])

    print('OK', end='\n\n')

    # Create simulator (agents' manager)
    print('Simulating distribution...', end=' ')

    simulator = FeatureDistributed.load(data, classifiers, p['overlap'], p['test_size'])

    print('OK', end='\n\n')

    ###########################################################################
    # CROSS VALIDATION ########################################################
    ###########################################################################
    print('Cross validating...', end=' ')

    scores = simulator.cross_validate(p['k_fold'], scorers, p['iterations'])

    print('OK', end='\n\n')

    ###########################################################################
    # TEST ####################################################################
    ###########################################################################
    print('Testing models...', end=' ')

    score = simulator.predict_score(scorers)

    print('OK', end='\n\n')

    # Save scores
    print('Saving scores...', end=' ')

    names = list(p['classifiers'].keys())
    n = len(names)

    [scores[i].to_csv('{}/cv_scores_{}.csv'.format(args.params_folder, names[i])) for i in range(n)]

    # Save test scores
    score.index = names    # line names as classifers' names
    score.to_csv('{}/test_scores.csv'.format(args.params_folder))

    print('OK', end='\n\n')
