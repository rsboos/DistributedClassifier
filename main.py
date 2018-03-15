import json
import argparse
import numpy as np

from pandas import DataFrame, concat
from src.agents import Voter, Combiner, Arbiter, Mathematician
from src.selectors import MetaDiff, MetaDiffInc, MetaDiffIncCorr
from src.test import test, load_imports, split_parts, load_scorers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--params",
                        dest="params_folder",
                        help="Folder where a file params.json is.",
                        required=True)

    # Validate params
    args = parser.parse_args()

    # Load params
    params = open("{}/params.json".format(args.params_folder), 'r')
    p = json.load(params)

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
         results_path=args.params_folder)
