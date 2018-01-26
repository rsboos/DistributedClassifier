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
test_size = 0.3
iterations = 10
class_column = -1
results = 'tests/example/'
dataset = 'datasets/seed.csv'
aggr = Voter(['borda', 'copeland', 'dowdall', 'plurality'])

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

###############################################################################
# Application
###############################################################################
# Load data
data = Data.load(dataset, class_column)

# Simulate distribution
classif_call = list(classifiers.values())
simulator = FeatureDistributed.load(data, classif_call, aggr, overlap, test_size)

# Run Cross-Validation
ranks, c_scores, r_scores = simulator.cross_validate(k_fold, scorers, iterations)

# Run tests
simulator.fit()
test_ranks, test_cscores, test_rscores = simulator.predict(scorers)

# Here, we had evaluated the model with social choice functions, but you can
# use the simulator model to predict using your own testset, like:
#
# test_ranks, test_scores = simulator.predict(['borda'], scorers, testset)
#

###############################################################################
# Save results
###############################################################################
# Save CV scores
names = list(classifiers.keys())
n = len(names)
[c_scores[i].to_csv('{}/cv_scores_{}.csv'.format(results, names[i])) for i in range(n)]

if type(aggr) is Voter:
    aggr_names = aggr.methods
elif type(aggr) is Combiner:
    aggr_names = [aggr.name]
elif type(aggr) is Arbiter:
    aggr_names = ['arb']
else:
    aggr_names = []

n = len(aggr_names)
[r_scores[i].to_csv('{}/cv_scores_{}.csv'.format(results, aggr_names[i])) for i in range(n)]

# Save rankings
for k in ranks:
    n_rank = list(map(lambda x: data.map_classes(x), ranks[k]))
    DataFrame(n_rank).to_csv('{}/cv_ranks_{}.csv'.format(results, k))

# Save test scores
for k in test_ranks:
    test_ranks[k] = data.map_classes(test_ranks[k])

test_ranks = DataFrame(test_ranks).T
test_cscores = DataFrame(test_cscores).T
test_rscores = DataFrame(test_rscores).T

test_cscores.index = names
test_scores = concat([test_cscores, test_rscores], keys=['classifiers', 'aggregators'], axis=0, copy=False)

test_ranks.to_csv('{}/test_ranks.csv'.format(results))
test_scores.to_csv('{}/test_scores.csv'.format(results))

# Save models
n = len(names)

for i in range(n):
    learner = simulator.learners[i]
    joblib.dump(learner.classifier, '{}/model_{}.pkl'.format(results, names[i]))

# Create CV summary
stats = summary(c_scores + r_scores)             # create summary
stats.index = names + aggr_names                 # line names as social choice functions' names
stats.to_csv('{}/cv_summary.csv'.format(results))
