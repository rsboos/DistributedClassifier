from pandas import read_csv, DataFrame, concat
from glob import glob
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.metrics import accuracy_score, precision_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


classifiers = { "gnb": GaussianNB(),
                "svc": SVC(probability=True),
                "mlp": MLPClassifier(),
                "dtree": DecisionTreeClassifier(),
                "knn": KNeighborsClassifier()}

scoring = { "f1_macro": make_scorer(f1_score, average='macro'),
            "f1_micro": make_scorer(f1_score, average='micro'),
            "recall_macro": make_scorer(recall_score, average='macro'),
            "recall_micro": make_scorer(recall_score, average='micro'),
            "accuracy": make_scorer(accuracy_score),
            "precision_macro": make_scorer(precision_score, average='macro'),
            "precision_micro": make_scorer(precision_score, average='micro')}

for dataset_path in glob('data/*'):

    data = read_csv(dataset_path)
    data = data.values

    X = data[:, :-1]
    y = data[:, -1]

    for i in range(10):
        skf = KFold(10, True, i)

        scores = {'mean': {}, 'std': {}}
        for name, classifier in classifiers.items():
            folds = skf.split(X, y)

            scores['mean'].setdefault(name, {})
            scores['std'].setdefault(name, {})

            try:
                cv_scores = cross_validate(classifier, X, y, cv=folds, scoring=scoring)
            except np.linalg.linalg.LinAlgError:
                continue

            for metric in scoring.keys():
                metric_score = cv_scores['test_' + metric]

                scores['mean'][name].setdefault(metric, [])
                scores['std'][name].setdefault(metric, [])

                scores['mean'][name][metric].append(metric_score.mean())
                scores['std'][name][metric].append(metric_score.std())

    for k in scores.keys():
        for classifier in scores[k].keys():
            for metric in scores[k][classifier].keys():
                if k == 'mean':
                    scores[k][classifier][metric] = np.mean(scores[k][classifier][metric])
                else:
                    scores[k][classifier][metric] = np.std(scores[k][classifier][metric])

    dataset_name = dataset_path.split('/')[-1]

    mean = DataFrame(scores['mean']).T
    std = DataFrame(scores['std']).T
    df = concat([mean, std], keys=['mean', 'std'], axis=1, copy=False)

    df.to_csv('results/' + dataset_name)
