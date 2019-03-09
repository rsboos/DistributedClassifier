import argparse

from src.cluster_analysis import ClusterAnalysis
from src.regression import RegressionAnalysis
from src.classification import ClassificationAnalysis
from src.graphics import Boxplot, NewickTree, GGPlot, Histogram


def main(args):
    regression = 'regression'
    classification = 'classification'

    analysis_data = 'data/datasets.csv'
    evaluation_path = '../evaluation/tests'

    overlap = 0

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
    clusters = ClusterAnalysis.clusters()

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

        graphic.overlap_performance()
        graphic.overlap_type_performance()

        graphic.performance(overlap=overlap)
        graphic.save('bp-performance-{}.pdf'.format(str(overlap)))

        graphic.baselined_performance(overlap=overlap)
        graphic.save('bp-baseline-performance-{}.pdf'.format(str(overlap)))

        graphic.type_performance(overlap=overlap)
        graphic.save('bp-type-performance-{}.pdf'.format(str(overlap)))

        graphic.baselined_type_performance(overlap=overlap)
        graphic.save('bp-baseline-type-performance-{}.pdf'.format(str(overlap)))

        graphic.dataset_performance(overlap=overlap)
        graphic.baselined_dataset_performance(overlap=overlap)

        graphic.cluster_performance(overlap=overlap)
        graphic.baselined_cluster_performance(overlap=overlap)

        graphic.dataset_method_performance(overlap=overlap)
        graphic.baselined_dataset_method_performance(overlap=overlap)

        for name, cluster in clusters.items():
            graphic = Boxplot()

            graphic.overlap_performance(cluster, name)
            graphic.overlap_type_performance(cluster, name)

            graphic.performance(cluster, overlap)
            graphic.save('bp-performance-{}-{}.pdf'.format(name, str(overlap)))

            graphic.baselined_performance(cluster, overlap)
            graphic.save('bp-baseline-performance-{}-{].pdf'.format(name, str(overlap)))

            graphic.type_performance(cluster, overlap)
            graphic.save('bp-type-performance-{}-0.pdf'.format(name, str(overlap)))

            graphic.baselined_type_performance(cluster, overlap)
            graphic.save('bp-baseline-type-performance-{}-{}.pdf'.format(name, str(overlap)))

        if args.show:
            graphic.show()
    elif args.ggplot:
        ggplot = GGPlot()
        ggplot.dataset_by_methods(overlap)
    elif args.hist:
        hist = Histogram()
        hist.feature_by_cluster()

    if args.newick is not None:
        graphic = NewickTree()
        graphic.create(args.newick)

    if args.cluster_analysis is not None:
        analysis = ClusterAnalysis()
        analysis.process()


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

    parser.add_argument("-c", "--cluster-analysis",
                        dest="cluster_analysis",
                        default=None,
                        help="Make a feature analysis by each cluster.")

    parser.add_argument("-gg", "--ggplot",
                        dest="ggplot",
                        default=None,
                        choices=['method-dataset'],
                        help="Create a specified graphic.")

    parser.add_argument("-hst", "--hist",
                        dest="hist",
                        default=None,
                        help="Create a specified graphic.")

    args = parser.parse_args()
    main(args)
