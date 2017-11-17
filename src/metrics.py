import numpy as np

from pandas import DataFrame
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
        del score['fit_time']
        del score['score_time']

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

###############################################################################
################################## SCORERS ####################################
###############################################################################
def accuracy():
    """Return a scorer for accuracy."""
    return met.accuracy_score


def auc():
    """Return a scorer for AUC."""
    return met.auc


def precision():
    """Return a scorer for precision."""
    return met.average_precision_score


def f1_score():
    """Return a scorer for F-score."""
    return met.f1_score


def recall():
    """Return a scorer for recall."""
    return met.recall_score


def sensitivity_score(y_true, y_pred, **kwargs):
    """Return a sensitivity score (true positive rate)."""
    cm = met.confusion_matrix(y_true, y_pred)

    tp = np.diag(cm).sum()              # sum of diagonal elements
    fn = cm.sum(axis=1).sum() - tp      # sum of all lines - diagonal

    return tp / (tp + fn)


def sensitivity():
    """Return a scorer for sensitivity."""
    return met.make_scorer(sensitivity_score)


def specificity_score(y_true, y_pred, **kwargs):
    """Return a specificity score (true negative rate)."""
    cm = met.confusion_matrix(y_true, y_pred)

    tp = np.diag(cm).sum()              # sum of diagonal elements
    fn = cm.sum(axis=1).sum() - tp      # sum of all lines - diagonal
    fp = cm.sum(axis=0).sum() - tp      # sum of all columns - diagonal
    tn = cm.sum() - (fp + fn + tp)      # sum of all - (fp + fn + tp)

    return tn / (tn + fp)


def specificity():
    """Return a scorer for specificity."""
    return met.make_scorer(specificity_score)
