import argparse


def main(scorers, classifiers, args):
    pass


if __name__ == "__main__":
    # def abbreviations
    scorers = {
        'ac': 'sklearn.metrics.accuracy_score',
        'rc': 'sklearn.metrics.recall_score',
        'pr': 'sklearn.metrics.precision_score',  # check for correct name
        'au': 'sklearn.metrics.auc',
        'f1': 'sklearn.metrics.f1_score',
        'sn': 'metrics.sensitivity',
        'sp': 'metrics.specificity'
    }

    classifiers = {
        # Binary classification


        # Inherently multiclass:
        'mcbnb': 'sklearn.naive_bayes.BernoulliNB',
        'mcdtc': 'sklearn.tree.DecisionTreeClassifier',
        'mcetc': 'sklearn.tree.ExtraTreeClassifier',
        'mcets': 'sklearn.ensemble.ExtraTreesClassifier',
        'mcgnb': 'sklearn.naive_bayes.GaussianNB',
        'mcknc': 'sklearn.neighbors.KNeighborsClassifier',
        'mclsv': 'sklearn.svm.LinearSVC',
        'mclrg': 'sklearn.linear_model.LogisticRegression',
        'mclrc': 'sklearn.linear_model.LogisticRegressionCV',
        'mcmlp': 'sklearn.neural_network.MLPClassifier',
        'mcrfc': 'sklearn.ensemble.RandomForestClassifier',

        #Multiclass as One-Vs-One:
        'oosvc': 'sklearn.svm.SVC',
        'oogpc': 'sklearn.gaussian_process.GaussianProcessClassifier',

        # Multiclass as One-Vs-All:
        'oagbc': 'sklearn.ensemble.GradientBoostingClassifier',
        'oagap': 'sklearn.gaussian_process.GaussianProcessClassifier',
        'oalsa': 'sklearn.svm.LinearSVC',
        'oalrg': 'sklearn.linear_model.LogisticRegression',
        'oalrc': 'sklearn.linear_model.LogisticRegressionCV',

        # Support multilabel:
        'mldtc': 'sklearn.tree.DecisionTreeClassifier',
        'mletc': 'sklearn.tree.ExtraTreeClassifier',
        'mlets': 'sklearn.ensemble.ExtraTreesClassifier',
        'mlknc': 'sklearn.neighbors.KNeighborsClassifier',
        'mlrfc': 'sklearn.ensemble.RandomForestClassifier',

        # Support multiclass-multioutput:
        'modtc': 'sklearn.tree.DecisionTreeClassifier',
        'moetc': 'sklearn.tree.ExtraTreeClassifier',
        'moets': 'sklearn.ensemble.ExtraTreesClassifier',
        'moknc': 'sklearn.neighbors.KNeighborsClassifier',
        'morfc': 'sklearn.ensemble.RandomForestClassifier'
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset",
                        dest="datasetpath",
                        help="Path to dataset.",
                        required=True)

    parser.add_argument("-c", "--class-column",
                        dest="class_column",
                        default=-1,
                        help="Class' column in dataset.")

    parser.add_argument("-i", "--iterations",
                        dest="n_it",
                        default=10,
                        help="Number of k-fold CV iterations.")

    parser.add_argument("-f", "--folds",
                        dest="n_folds",
                        default=10,
                        help="Number of folds for CV.")

    parser.add_argument("-t", "--test",
                        dest="test",
                        default=20,
                        help="Percent of test instances.")

    parser.add_argument("-o", "--overlap",
                        dest="overlap",
                        default=0,
                        help="Percent of overlaped features.")

    parser.add_argument("-s", "--scorers",
                        dest="scorers",
                        help="Scorers 2-letter-abbreviation. {}. Example for accuracy and precision: ac.pr".format(str(scorers)),
                        default=list(),
                        choices=list(scorers.keys()))

    parser.add_argument("-l", "--classifiers",
                        dest="classifiers",
                        help="Classifiers 5-letter-abbreviation. {}. Example for DecisionTree and LinearSVC: mclsv.mcdtc".format(str(classifiers)),
                        required=True,
                        choices=list(classifiers.keys()))

    args = parser.parse_args()
    main(scorers, classifiers, args)
