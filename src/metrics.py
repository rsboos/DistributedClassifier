import numpy as np

from pandas import DataFrame, concat
from sklearn import metrics as met


def cv_score(scores):
    """Build a score matrix where a fold is a line and a scorer a column.

    Keyword arguments:
        scores -- a list of scores from cross_validate's sklearn function
    """
    # Initialize structure
    score_matrix = dict()

    # For each iteration of CV...
    for score in scores:

        # Delete unwanted data
        try:
            del score['fit_time']
            del score['score_time']
        except KeyError:
            pass

        # For each test score...
        for key, values in score.items():

            # Filter scores. Get just the test scores
            if key.startswith('test_'):
                # Delete test_ prefix
                scorer = key.replace('test_', '')

                # Append scores to the scorer
                arr = score_matrix.get(scorer, [])
                score_matrix[scorer] = np.append(arr, values)

    # Convert dict to DataFrame where
    # keys are columns
    return DataFrame.from_dict(score_matrix)


def summary(scores):
    """Average scores from Cross-Validation.

    Keyword arguments:
        scores -- DataFrame or list of DataFrames
    """
    # Make score as list of DataFrames
    scores = scores if type(scores) is list else [scores]

    # Mean matrix
    means = dict()
    stds = dict()

    # For each CV scores, calculate mean...
    for i in range(len(scores)):
        score = scores[i]
        columns = score.columns

        # ... and append to mean matrix
        mean = score.mean(0)
        std = score.std(0)

        means[i] = list(mean.iloc[:])
        stds[i] = list(std.iloc[:])

    mean = DataFrame.from_dict(means).T
    mean.columns = columns

    std = DataFrame.from_dict(stds).T
    std.columns = columns

    return concat([mean, std], keys=['mean', 'std'], axis=1, copy=False)

###############################################################################
################################## SCORERS ####################################
###############################################################################
def confusion_matrix(y_true, y_pred, **kwargs):
    """Return the confusion matrix according to a label order."""
    labels = kwargs.get('labels', list(set(y_true)))
    labels = sorted(range(len(labels)), key=lambda k: labels[k])

    return met.confusion_matrix(y_true, y_pred, labels=labels)


def sensitivity_score(y_true, y_pred, **kwargs):
    """Return a sensitivity score (true positive rate)."""
    cm = confusion_matrix(y_true, y_pred, **kwargs)

    tp = cm[0,0]
    fn = cm[0,1]

    return tp / (tp + fn)


def specificity_score(y_true, y_pred, **kwargs):
    """Return a specificity score (true negative rate)."""
    cm = confusion_matrix(y_true, y_pred, **kwargs)

    tp = cm[0,0]
    fn = cm[0,1]
    fp = cm[1,0]
    tn = cm[1,1]

    return tn / (tn + fp)
