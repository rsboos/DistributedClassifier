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

###############################################################################
################################## SCORERS ####################################
###############################################################################
def sensitivity_score(y_true, y_pred, **kwargs):
    """Return a sensitivity score (true positive rate)."""
    cm = met.confusion_matrix(y_true, y_pred)

    tp = cm[1,1]
    fn = cm[1,0]

    return tp / (tp + fn)


def specificity_score(y_true, y_pred, **kwargs):
    """Return a specificity score (true negative rate)."""
    cm = met.confusion_matrix(y_true, y_pred)

    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]

    return tn / (tn + fp)

f1 = met.make_scorer(met.f1_score)
auc = met.make_scorer(met.roc_auc_score)
recall = met.make_scorer(met.recall_score)
accuracy = met.make_scorer(met.accuracy_score)
precision = met.make_scorer(met.precision_score)
sensitivity = met.make_scorer(sensitivity_score)
specificity = met.make_scorer(specificity_score)
