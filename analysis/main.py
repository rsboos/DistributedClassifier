import argparse
from src.regression import RegressionAnalysis, TreeAnalysis


def main(args):
    regression = 'regression'
    classificaion = 'classification'

    analysis_data = 'data/datasets.csv'
    evaluation_path = '../evaluation/tests'

    if args.all == regression:
        regression = RegressionAnalysis()
        regression.process(analysis_data, evaluation_path)
        # regression.evaluate()

        TreeAnalysis.grow_trees()
        TreeAnalysis.get_important_nodes(analysis_data)

    elif args.process == regression:
        regression = RegressionAnalysis()
        regression.process(analysis_data, evaluation_path)

    elif args.evaluate == regression:
        regression = RegressionAnalysis()
        regression.evaluate()

    elif args.trees == regression:
        TreeAnalysis.grow_trees()

    elif args.important_nodes == regression:
        TreeAnalysis.get_important_nodes(analysis_data)


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

    parser.add_argument("-i", "--get-important-nodes",
                        dest="important_nodes",
                        default=None,
                        help="Only for regression. "
                             "Extract important nodes from trees.")

    parser.add_argument("-a", "--all",
                        dest="all",
                        default=None,
                        help="Make a pipeline with all analysis for "
                             "regression or classification.")

    args = parser.parse_args()
    main(args)
