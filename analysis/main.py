import argparse

from src.graphics import Boxplot, NewickTree
from src.regression import RegressionAnalysis
from src.classification import ClassificationAnalysis


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
    clusters = {
        'cluster1': ['yeast', 'vowel', 'g1318-95-9', 'cancer', 'sharkattack', 'iris', 'soybean', 'segmentation', 'glass'],
        'cluster2': ['g2827-95-4', 'optdigits', 'sat', 'crack', 'heroine', 'amyl', 'katemine', 'vsa', 'meth', 'caff',
                     'seed', 'unbalanced1140-68-6', 'lsd', 'legalh', 'amphet', 'benzos', 'mushrooms', 'coke', 'ecstasy',
                     'alcohol', 'cannabis', 'nicotine'],
        'cluster3': ['unbalanced9636-40-6', 'diabetics', 'brokenmachine', 'unbalanced8725-53-8'],
        'cluster4': ['unbalanced5627-61-6', 'pageblocks', 'activityrecog', 'motionsense', 'pendigits', 'letter',
                     'unbalanced5394-89-5', 'banking', 'g8848-86-2', 'g5946-47-2', 'g6576-46-2']
    }

    if args.graphics == 'bp-ranking':
        graphic = Boxplot()
        graphic.ranking()
        graphic.save('bp-ranking.pdf')

        graphic.type_ranking()
        graphic.save('bp-type-ranking.pdf')

        for name, cluster in clusters.items():
            graphic = Boxplot()
            graphic.ranking(cluster)
            graphic.save('bp-ranking-{}.pdf'.format(name))

            graphic.type_ranking(cluster)
            graphic.save('bp-type-ranking-{}.pdf'.format(name))

        if args.show:
            graphic.show()
    elif args.graphics == 'bp-performance':
        graphic = Boxplot()
        graphic.performance()
        graphic.save('bp-performance.pdf')

        graphic.type_performance()
        graphic.save('bp-type-performance.pdf')

        for name, cluster in clusters.items():
            graphic = Boxplot()
            graphic.performance(cluster)
            graphic.save('bp-performance-{}.pdf'.format(name))

            graphic.type_performance(cluster)
            graphic.save('bp-type-performance-{}.pdf'.format(name))

        if args.show:
            graphic.show()

    if args.newick is not None:
        graphic = NewickTree()
        graphic.create(args.newick)

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
                        choices=['bp-ranking', 'bp-performance'],
                        help="Create a specified graphic.")

    parser.add_argument("-n", "--newick",
                        dest="newick",
                        default=None,
                        choices=['ward', 'average', 'complete'],
                        help="Display a Newick Tree.")

    parser.add_argument("-s", "--show",
                        dest="show",
                        default=False,
                        choices=['true', 'false'],
                        help="Show or not a graphic.")

    args = parser.parse_args()
    main(args)
