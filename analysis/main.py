import argparse

from src.graphics import Boxplot
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
    if args.process == regression:
        regression = RegressionAnalysis()
        regression.process(analysis_data, evaluation_path)

    elif args.evaluate == regression:
        regression = RegressionAnalysis()
        regression.evaluate()

    elif args.trees == regression:
        RegressionAnalysis.grow_trees()

    elif args.important_nodes == regression:
        RegressionAnalysis.get_important_nodes(analysis_data)

    # Classification
    elif args.process == classification:
        classification = ClassificationAnalysis()
        classification.process(analysis_data, evaluation_path)

    elif args.evaluate == classification:
        classification = ClassificationAnalysis()
        classification.evaluate()

    elif args.trees == classification:
        ClassificationAnalysis.grow_trees()

    elif args.important_nodes == classification:
        ClassificationAnalysis.get_important_nodes(analysis_data)

    # Graphics
    if args.graphics == 'boxplot':
        graphic = Boxplot()
        graphic.save('boxplot.png')

        if args.show:
            graphic.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--process",
                        dest="process",
                        default=None,
                        choices=['regression', 'classification'],
                        help="Create data sets for evaluation.")

    parser.add_argument("-e", "--evaluate",
                        dest="evaluate",
                        default=None,
                        choices=['regression', 'classification'],
                        help="Type of evaluation (regression or "
                             "classification). Evaluate data sets.")

    parser.add_argument("-t", "--make-trees",
                        dest="trees",
                        default=None,
                        choices=['regression', 'classification'],
                        help="Create trees from DecisionTree's algorithm.")

    parser.add_argument("-i", "--get-important-nodes",
                        dest="important_nodes",
                        default=None,
                        choices=['regression'],
                        help="Extract important nodes from trees.")

    parser.add_argument("-g", "--graphics",
                        dest="graphics",
                        default=None,
                        choices=['boxplot'],
                        help="Create a specified graphic.")

    parser.add_argument("-s", "--show",
                        dest="show",
                        default=False,
                        choices=['true', 'false'],
                        help="Show or not a graphic.")

    args = parser.parse_args()
    main(args)
