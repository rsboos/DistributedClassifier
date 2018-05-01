from .split import P3StratifiedKFold, Distributor
from .metrics import cv_score, score
from .agents import Learner
from .data import Data


class FeatureDistributedSimulator():
	"""Feature distributed classification learning simulator.

	Description:
		This class simulates a distributed learning using classifier agents. It divides
		the data vertically, i. e., it divides the features randomly between the learners.
	"""
	def __init__(self, classifiers, agreggators):
		"""Set private properties.

		Keyword arguments:
			classifiers -- a list of classifiers' instances
			aggregators -- a list of agreggators' inatances
		"""
		self.__classifiers = classifiers
		self.__aggregators = agreggators

	def repeated_cv(self, data, overlap, scoring={}, random_state=None, n_it=10):
		"""Runs the cross_validate function for each agent and returns a list with each learner's scores

		Keyword arguments:
			data -- a Data object
			classifiers -- a list of classifiers, len(classifiers) define the number of learners
			overlap -- if float, should be between 0.0 and 1.0 and represents the percentage
					   of parts' in common. If int, should be less than or equal to the
					   number of features/instances and represents the number of common
					   features/instances. If list, represents the features'/instances'
					   indexes. By default, the value is set to 0.
			scoring -- metrics to be returned (default {})*
			random_state -- int, RandomState instance or None, optional, default=None
		        If int, random_state is the seed used by the random number generator;
		        If RandomState instance, random_state is the random number generator;
		        If None, the random number generator is the RandomState instance used
		        by `np.random`. Used when ``shuffle`` == True.
			n_it -- number of cross-validation iterations (default 10, i. e., 10 k-fold cross-validation)

		For how to use scoring:
		http://scikit-learn.org/stable/modules/cross_validation.html
		"""
		# Number of folds
		k_fold = 10

		# Initializes an empty dict for scores and rankings
		scores = dict()
		ranks = dict()

		# Splits into k training and test folds for cross-validation
		skf = P3StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)

		for seed in range(n_it):
			learners = self.__distribute(data, overlap, seed)
			n = len(learners)

			# Gets a sample of the data for the splitter
			sample_x = learners[0].dataset.x  # instances
			sample_y = learners[0].dataset.y  # classes

			# Create folds and iterate
			for train_i, val_i, test_i in skf.split(sample_x, sample_y):
				# Initialize empty list for probabilities
				combiner_input = list()
				probabilities = list()
				predictions = list()

				# For each learner...
				for j in range(n):
					# Compute a fold
					_, proba = learners[j].run_fold(train_i, val_i, scoring)

					# Save for combiner
					combiner_input.append(proba)

					# Get test data
					test_x = learners[j].dataset.x[test_i, :]
					test_y = learners[j].dataset.y[test_i]

					# Get model's predictions
					predi = learners[j].predict(test_x)
					proba = learners[j].predict_proba(test_x)

					# Get scores
					metrics = score(test_y, predi, scoring)

					# Save predictions and probabilities
					predictions.append(predi)
					probabilities.append(proba)

					# Save score
					scores.setdefault(j, [])
					scores[j].append(metrics)

				# Aggregate probabilities with different methods
				aggr_r, aggr_s = {}, {}

				for k in range(len(self.__aggregators)):
					rank, metrics = self.__aggregators[k].aggr(y_true=sample_y[test_i],
				                                 			   y_pred=predictions,
				                                 			   y_proba=probabilities,
				                                 			   x=combiner_input,
												 			   y=sample_y[val_i],
												 			   testset=probabilities,
				                                 			   learners=learners,
												 			   test_i=test_i,
				                                 			   scoring=scoring)

					aggr_r.update(rank)
					aggr_s.update(metrics)

				# Save ranks
				for k in aggr_r:
					ranks.setdefault(k, [])
					ranks[k].append(aggr_r[k])

				# Save scores
				for k in aggr_s:
					scores.setdefault(k, [])
					scores[k].append(aggr_s[k])

		# Return the ranks and aggregated scores as DataFrames for each learner
		return ranks, [cv_score(scores[k]) for k in scores]

	def __distribute(self, data, overlap, random_state):
		learners = []
		n_learners = len(self.__classifiers)

		distributor = Distributor(n_learners, overlap, random_state)

		indexes = distributor.split(data)
		n_indexes = len(indexes)

		for i in range(n_indexes):
			features = indexes[i]

			dataset = Data(data.x[:, features], data.y)
			classifier = self.__classifiers[i]

			learner = Learner(dataset, classifier)
			learners.append(learner)

		return learners
