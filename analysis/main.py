import argparse
from src.regression import RegressionAnalysis, TreeAnalysis


def main(args):
    regression = 'regression'
    classificaion = 'classification'

    if args.process == regression:
        regression = RegressionAnalysis()
        regression.process('data/datasets.csv', '../evaluation/tests')

    elif args.evaluate == regression:
        regression = RegressionAnalysis()
        regression.evaluate()

    elif args.trees == regression:
        TreeAnalysis.grow_trees()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--process",
                        dest="process",
                        default=None,
                        help="Type of process (regression or classification). "
                             "Create data sets for evaluation.")

    parser.add_argument("-e", "--evaluate",
                        dest="evaluate",
                        default=None,
                        help="Type of evaluation (regression or "
                             "classification). Evaluate data sets.")

    parser.add_argument("-t", "--make-trees",
                        dest="trees",
                        default=None,
                        help="Only for regression. "
                             "Create trees from DecisionTreeRegressor.")

    args = parser.parse_args()
    main(args)
