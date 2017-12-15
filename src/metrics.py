import numpy as np

from math import fsum
from sklearn import metrics as met
from pandas import DataFrame, concat


def cv_score(scores):
    """Build a score matrix where a fold is a line and a scorer a column.

    Keyword arguments:
        scores -- a list of scores from cross_validate's sklearn function
    """
    # Initialize structure
    score_matrix = dict()

    # For each iteration of CV...
    for score in scores:

        # For each test score...
        for key, values in score.items():
            # Append scores to the scorer
            arr = score_matrix.get(key, [])
            score_matrix[key] = np.append(arr, values)

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


def join_ranks(rankings):
    """Receive one rank per class and join them by score.

    Keyword arguments:
        rankings -- a list of class ranks

    Return: a list with classes.
    """
    classes = list()                # dict of classes' indexes by scf
    n_ranks = len(rankings)         # # of classes = # of ranks
    rank_size = len(rankings[0])    # # of instances = length of a rank

    # For each instance i...
    for i in range(rank_size):
        b = 0  # 0 = class with the biggest value

        # For each class rank, get class with highest score
        for j in range(1, n_ranks):
            b = j if rankings[j][i][1] > rankings[b][i][1] else b

        # Save class
        classes.append(b)

    return classes


###############################################################################
################################## SCORERS ####################################
###############################################################################
def score(y_true, y_pred, scoring):
    """Calculate predictions' metrics and return a dict with metrics.

    Keyword arguments:
        y_true -- true data
        y_pred -- predicted data
        scoring -- a dict as {<score name>: <scorer func>}
    """
    # Init dict
    scores = dict()

    # For each metric...
    for k in scoring:
        # Get the scorer function
        scorer = scoring[k]

        # Calculate and save score
        scores[k] = scorer._score_func(y_true, y_pred)

    return scores


def confusion_matrix(y_true, y_pred, **kwargs):
    """Return the confusion matrix according to a label order.

    Keyword arguments:
        y_true -- true values
        y_pred -- prediction values
        labels -- list of labels to index matrix (default set(y_true))
    """
    labels = kwargs.get('labels', list(set(y_true)))
    labels = sorted(range(len(labels)), key=lambda k: labels[k])

    return met.confusion_matrix(y_true, y_pred, labels=labels)


def macro(a):
    """Calculate the arithmetic mean.

    Keyword arguments:
        a -- a list of numbers
    """
    n = len(a)
    return fsum(a) / float(n)


def micro(a, b):
    """Calculate the micro average.

    Keyword arguments:
        values -- a list of numbers
    """
    return fsum(a) / fsum(b)


def average(values, avg, func):
    """Calculate average according to type.

    Keyword arguments:
        values -- a list of tuples
        avg -- {'macro', 'micro'}
        func -- score function as func(a, b) -> c
    """
    if avg == 'macro':
        # Calculate scores using func
        scores = list(map(lambda x: func(x[0], x[1]), values))

        # Return macro avg
        return macro(scores)
    elif avg == 'micro':
        # Separate values
        a, b = zip(*values)
        a, b = list(a), list(b)

        # Return micro avg
        return micro(a, a + b)
    else:
        raise ValueError('{} is not a valid average.'.format(avg))

def recall_score(y_true, y_pred, **kwargs):
    """Return a sensitivity score (true positive rate).

    Keyword arguments:
        y_true -- true values
        y_pred -- prediction values
        labels -- list of labels to index matrix (default set(y_true))
        average -- {(default 'macro'), 'micro'}. For multiclass, only.
    """
    return sensitivity_score(y_true, y_pred, **kwargs)

def sensitivity_score(y_true, y_pred, **kwargs):
    """Return a sensitivity score (true positive rate).

    Keyword arguments:
        y_true -- true values
        y_pred -- prediction values
        labels -- list of labels to index matrix (default set(y_true))
        average -- {(default 'macro'), 'micro'}. For multiclass, only.
    """
    cm = confusion_matrix(y_true, y_pred, **kwargs)  # confusion matrix
    sens = lambda tp, fn: tp / (tp + fn)             # sensitivity function
    n, _ = cm.shape                                  # matrix dimension = # classes

    # If it is binary...
    if n == 2:
        # Return TP / (TP + FN)
        return sens(cm[0, 0], cm[0, 1])
    else:
        # Average type
        avg = kwargs.get('average', 'macro')

        # (TP, FN) list
        values = [(cm[i, i], cm.sum(axis=1)[i] - cm[i, i]) for i in range(n)]

        # Return calculated avg
        return average(values, avg, sens)

def specificity_score(y_true, y_pred, **kwargs):
    """Return a specificity score (true negative rate).

    Keyword arguments:
        y_true -- true values
        y_pred -- prediction values
        labels -- list of labels to index matrix (default set(y_true))
        average -- {(default 'macro'), 'micro'}. For multiclass, only.
    """
    cm = confusion_matrix(y_true, y_pred, **kwargs)  # confusion matrix
    spec = lambda tn, fp: tn / (tn + fp)             # sensitivity function
    n, _ = cm.shape                                  # matrix dimension = # classes

    # If it is binary...
    if n == 2:
        # Return TN / (TN + FP)
        return spec(cm[1,1], cm[1,0])
    else:
        # Average type
        avg = kwargs.get('average', 'macro')

        # (TN, FP) list
        values = list()

        for i in range(n):
            tp = cm[i, i]                       # true positive
            fn = cm.sum(axis=1)[i] - tp         # false negative
            fp = cm.sum(axis=0)[i] - tp         # false positive
            tn = cm.sum() - (tp + fn + fp)      # true negative
            values.append((tn, fp))             # save (tn, fp)

        # Return calculated avg
        return average(values, avg, spec)
