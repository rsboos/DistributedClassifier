import json
import argparse

from pandas import DataFrame, concat
from src.data import Data
from src.metrics import summary
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
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

    print('OK')

    # Evaluate classifiers
    print('Loading classifiers...')

    classifiers = list()

    for c in p['classifiers'].values():                  # for each classfier
        parts = c.split('.')                             # separate in parts

        i = ['(' in part for part in parts].index(1)     # filter parts

        modules = '.'.join(parts[:i])                    # get modules
        classifier = '.'.join(parts[i:])                 # get classifier

        i = classifier.find('(')                         # filter classifier

        str_eval = "from {} import {}".format(modules, classifier[:i])

        print(str_eval)

        # Execute import
        exec(str_eval)

        # Evaluate a classifier
        classifiers.append(eval(classifier))

    print('Done.')

    # Evaluate metrics
    print('Loading metrics...')
    scorers = dict()

    for n, c in p['metrics'].items():                    # for each metric (name, method)
        parts = c.split('.')                             # separate in parts

        i = ['(' in part for part in parts].index(1)     # filter parts

        modules = '.'.join(parts[:i])                    # get modules
        metric = '.'.join(parts[i:])                     # get metric

        i = metric.find('(')                             # filter score func
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

    scf_name, scf_callable = zip(*p['social_functions'].items())
    scf_name, scf_callable = list(scf_name), list(scf_callable)

    print('Done.')

    ###########################################################################
    # SIMULATE DISTRIBUTION ###################################################
    ###########################################################################
    print('Loading dataset {}...'.format(p['dataset']), end=' ')

    data = Data.load(p['dataset'], p['class_column'])

    print('OK')

    # Create simulator (agents' manager)
    print('Simulating distribution...', end=' ')

    simulator = FeatureDistributed.load(data, classifiers, p['overlap'], p['test_size'])

    print('OK')

    ###########################################################################
    # CROSS VALIDATION ########################################################
    ###########################################################################
    print('Cross validating...', end=' ')

    ranks, classif_scores, rank_scores = simulator.cross_validate(scf_callable, p['k_fold'],
                                                                  scorers, p['iterations'])

    print('OK')

    ###########################################################################
    # TEST ####################################################################
    ###########################################################################
    print('Testing models...', end=' ')

    simulator.fit()
    test_ranks, test_cscores, test_rscores = simulator.predict(scf_callable, scorers)

    print('OK')

    ###########################################################################
    # SAVE RESULTS ############################################################
    ###########################################################################
    # Save CV scores
    print('Saving CV scores...', end=' ')

    names = list(p['classifiers'].keys())
    n = len(names)
    [classif_scores[i].to_csv('{}/cv_scores_{}.csv'.format(args.params_folder, names[i])) for i in range(n)]

    n = len(scf_name)
    [rank_scores[i].to_csv('{}/cv_scores_{}.csv'.format(args.params_folder, scf_name[i])) for i in range(n)]

    print('OK')

    # Save rankings
    print('Saving CV ranks...', end=' ')

    for scf in ranks:
        n_rank = list(map(lambda x: data.map_classes(x), ranks[scf]))
        DataFrame(n_rank).to_csv('{}/cv_ranks_{}.csv'.format(args.params_folder, scf))

    print('OK')

    # Save test scores
    print('Saving test ranks and scores...', end=' ')

    for scf in test_ranks:
        test_ranks[scf] = data.map_classes(test_ranks[scf])

    test_ranks = DataFrame(n_rank).T
    test_cscores = DataFrame(test_cscores).T
    test_rscores = DataFrame(test_rscores).T

    test_cscores.index = names
    test_scores = concat([test_cscores, test_rscores], keys=['classifiers', 'social_functions'], axis=0, copy=False)

    test_ranks.to_csv('{}/test_ranks.csv'.format(args.params_folder))
    test_scores.to_csv('{}/test_scores.csv'.format(args.params_folder))

    print('OK')

    # Save models
    print('Saving models...', end=' ')

    names = list(p['classifiers'].keys())
    n = len(names)

    for i in range(n):
        learner = simulator.learners[i]
        joblib.dump(learner.classifier, '{}/model_{}.pkl'.format(args.params_folder, names[i]))

    print('OK')

    # Create CV summary
    print('Creating CV summary...', end=' ')

    stats = summary(classif_scores + rank_scores)     # create summary
    stats.index = names + scf_name                    # line names as social choice functions' names
    stats.to_csv('{}/cv_summary.csv'.format(args.params_folder))

    print('OK')
