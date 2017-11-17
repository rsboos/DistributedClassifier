import numpy as np

from pandas import DataFrame


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
