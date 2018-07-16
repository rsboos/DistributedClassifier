import argparse
from src.tree_analysis import TreeAnalysis
from src.regression import RegressionAnalysis
from src.classification import ClassificationAnalysis
from src.path import RegressionPath, ClassificationPath


def main(args):
    regression = 'regression'
    classification = 'classification'

    analysis_data = 'data/datasets.csv'
    evaluation_path = '../evaluation/tests'

    # Regression
    if args.all == regression:
        regression = RegressionAnalysis()
        regression.process(analysis_data, evaluation_path)
        regression.evaluate()

        regression.grow_trees()
        TreeAnalysis.get_important_nodes(analysis_data, RegressionPath())

    elif args.process == regression:
        regression = RegressionAnalysis()
        regression.process(analysis_data, evaluation_path)

    elif args.evaluate == regression:
        regression = RegressionAnalysis()
        regression.evaluate()

    elif args.trees == regression:
        RegressionAnalysis.grow_trees()

    elif args.important_nodes == regression:
        TreeAnalysis.get_important_nodes(analysis_data, RegressionPath())

    # Classification
    elif args.process == classification:
        classification = ClassificationAnalysis()
        classification.process(analysis_data, evaluation_path)

    elif args.trees == classification:
        ClassificationAnalysis.grow_trees()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--process",
                        dest="process",
                        default=None,
                        help="Type of process (regression or __classification)."
                             "Create data sets for evaluation.")

    parser.add_argument("-e", "--evaluate",
                        dest="evaluate",
                        default=None,
                        help="Type of evaluation (regression or "
                             "__classification). Evaluate data sets.")

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
                             "regression or __classification.")

    args = parser.parse_args()
    main(args)
