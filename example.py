# Utilities
from src.data import Data
from pandas import DataFrame, concat
from sklearn.externals import joblib
from src.simulator import FeatureDistributed
from src.agents import Voter, Combiner, Arbiter

# Classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.metrics import make_scorer, recall_score, accuracy_score
from src.metrics import summary, specificity_score, sensitivity_score

###############################################################################
# Settings
###############################################################################
k_fold = 10
overlap = 0
iterations = 10
class_column = -1
results = 'tests/example/'
dataset = 'datasets/seed.csv'

classifiers = {
    'gaussian_nb': GaussianNB(),
    'svc': SVC(probability=True),
    'dtree': DecisionTreeClassifier(),
    'knn': KNeighborsClassifier()
}

scorers = {
    'auc': make_scorer(roc_auc_score),
    'f1': make_scorer(f1_score, average='binary'),
    'recall': make_scorer(recall_score),
    'specificity': make_scorer(specificity_score, labels=['Target', 'Non-Target']),
    'sensitivity': make_scorer(specificity_score, labels=['Target', 'Non-Target']),
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score)
}

voter = {
    "borda": "borda",
    "copeland": "copeland",
    "plurality": "plurality",
}

combiner = {
    'svc': SVC(probability=True),
    'dtree': DecisionTreeClassifier(),
}

###############################################################################
# Application
###############################################################################
# Load data
data = Data.load(dataset, class_column)

# Simulate distribution
classif_call = list(classifiers.values())
voter_call = Voter(list(voter.values()))
combiner_call = Combiner(list(combiner.values()))

simulator = FeatureDistributed.load(data, classif_call, overlap, voter=voter_call, combiner=combiner_call)

# Run Cross-Validation
ranks, c_scores, r_scores = simulator.cross_validate(k_fold, scorers, iterations)

###############################################################################
# Save results
###############################################################################
# Save CV scores
names = list(classifiers.keys())
n = len(names)
[c_scores[i].to_csv('{}/cv_scores_{}.csv'.format(results, names[i])) for i in range(n)]


voter_names = list(voter.keys())
combiner_names = list(combiner.keys())
aggr_names = voter_names + combiner_names

n = len(aggr_names)
[r_scores[i].to_csv('{}/cv_scores_{}.csv'.format(results, aggr_names[i])) for i in range(n)]

# Save rankings
for k in ranks:
    n_rank = list(map(lambda x: data.map_classes(x), ranks[k]))
    DataFrame(n_rank).to_csv('{}/cv_ranks_{}.csv'.format(results, k))

# Create CV summary
stats = summary(c_scores + r_scores)             # create summary
stats.index = names + aggr_names                 # line names as aggregators' names
stats.to_csv('{}/cv_summary.csv'.format(results))
