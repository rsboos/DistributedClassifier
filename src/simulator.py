import math
import copy
import numpy
import warnings

from .data import Data
from .split import P3StratifiedKFold
from .selectors import MetaDiffIncCorr
from .metrics import cv_score, score, join_ranks
from .agents import Learner, Voter, Combiner, Arbiter, Mathematician


class Simulator():
	"""Distributed classification learning simulator.

	Properties:
		learners -- a list of learners
		voter -- a Voter object
		combiner -- a Combiner object
	"""

	def __init__(self, **kwargs):
		"""Sets the properties, create the agents, divides the data between the agents.

		Keyword arguments:
			voter -- a Voter object with social choice functions (default None)
			combiner -- a Combiner object with a list of classifiers (default None)
			arbiter -- a Arbiter object
			mathematician -- a Mathematician object
		"""
		self.voter = kwargs.get('voter', Voter())
		self.arbiter = kwargs.get('arbiter', Arbiter(MetaDiffIncCorr))
		self.combiner = kwargs.get('combiner', Combiner())
		self.mathematician = kwargs.get('mathematician', Mathematician())

	@staticmethod
	def get_distribution(**kwargs):
		"""Not implemented. Should be implemented in a child class."""
		pass

	def repeated_cv(self, **kwargs):
		"""Not implemented. Should be implemented in a child class."""
		pass


class FeatureDistributed(Simulator):
	"""Feature distributed classification learning simulator.

	Description:
		This class simulates a distributed learning using classifier agents. It divides
		the data vertically, i. e., it divides the features randomly between the learners.

	Also see: Simulator class documentation.
	"""
	@classmethod
	def init_learners(cls, data, classifiers, overlap, random_state):
		# The number of classifiers define the number of learners
		n_learners = len(classifiers)

		# Gets the indexes to slice the data
		# Because the distribution is vertical, we can use a part
		# of the data with all features to get the slice indexes
		indexes = cls.get_distribution(data, n_learners, overlap, random_state)
		n_indexes = len(indexes)

		# Convert to ndarray for slicing
		data.x, data.y = numpy.array(data.x), numpy.array(data.y)

		learners = list()

		for i in range(n_indexes):
			features = indexes[i]								 # gets the features indexes

			dataset = Data(data.x[:, features], data.y)		 	 # gets the trainingset
			classifier = classifiers[i]			   			     # gets the classifier

			learner = Learner(dataset, classifier) 			     # creates the learner
			learners.append(learner)			   			     # saves the learner

		return learners

	@staticmethod
	def get_distribution(data, n, overlap=0, random_state=None):
		"""Vertically distributes the training data into n parts and returns a list of
		column indexes for each part

		Keyword arguments:
			data -- the training set to be splitted (Data)
			n -- number of divisions
			overlap -- if float, should be between 0.0 and 1.0 and represents the percentage
					   of parts' common features. If int, should be less than or equal to the
					   number of features and represents the number of common features. If list,
					   represents the features' columns indexes. By default, the value is set to 0.
			random_state -- int, RandomState instance or None, optional, default=None
		        If int, random_state is the seed used by the random number generator;
		        If RandomState instance, random_state is the random number generator;
		        If None, the random number generator is the RandomState instance used
		        by `np.random`. Used when ``shuffle`` == True.
		"""

		# Check the type of overlap variable and sets the n_common and common_features, where
		# n_common is the number of common features for each part and
		# common_features is a common features' index's list
		rs = numpy.random.RandomState(random_state)

		if type(overlap) is int:
			common_features = set(rs.choice(data.n_features, overlap))   # gets a n length list with random numbers between 0-n_features

		elif type(overlap) is float:
			n_common = math.ceil(data.n_features * overlap)			     # calculates the amount of features each part has in common
			common_features = set(rs.choice(data.n_features, n_common))  # gets a n length list with random numbers between 0-n_features

		elif type(overlap) is list:
			common_features = set(overlap)

		else: # is neither int, float and list
			raise TypeError("%s should be a float, an int or a list. %s given." % (str(overlap), type(overlap)))

		# Calculates the distinct features' indexes
		distinct_features = list(set(range(data.n_features)) - common_features) # common_features' complement

		# Just converts from set to list
		common_features = list(common_features)
		n_common = len(common_features)

		# Permutates the distinct features (randomize)
		distinct_features = list(rs.permutation(distinct_features))

		# Calculate the number of distinct features per part
		n_distinct = len(distinct_features)

		# Calculates the number of distinc features per part
		n_part_features = math.ceil(n_distinct / n)

		# A part should not have less than 2 features
		if n_part_features + n_common < 2:
			n = n_distinct / (2 - n_common)
			n_part_features = math.ceil(n_distinct / n)
			warnings.warn('Each division has less then 2 features. Narrowing down to {} divisions.' % n, FutureWarning)

		# Empty list for the loop
		distribution = []

		# The main object in the loop is the distinct_features list
		# We want to add a slice of this list with the common_features list
		# for each part
		for i in range(0, n_distinct + 1, n_part_features):
			j = i + n_part_features									# stop index for slice, garantees n features to the part
			part_indexes = common_features + distinct_features[i:j] # list of common and distinct indexes for the part
			distribution.append(part_indexes)						# adds the list to distribution

		# Returns the distribution list
		return distribution

	def repeated_cv(self, data, classifiers, overlap, scoring={}, random_state=None, n_it=10):
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

		for i in range(n_it):
			learners = cls.init_learners(data, classifiers, overlap, i)
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
				aggr_r, aggr_s = self.voter.aggr(y_true=sample_y[test_i],
				                                 y_pred=predictions,
				                                 y_proba=probabilities,
				                                 scoring=scoring)

				caggr_r, caggr_s = self.combiner.aggr(x=combiner_input,
													  y=sample_y[val_i],
													  testset=probabilities,
													  y_true=sample_y[test_i],
				                                 	  scoring=scoring)

				aaggr_r, aaggr_s = self.arbiter.aggr(learners=learners,
				                                     x=combiner_input,
				                                     y=sample_y[val_i],
				                                     y_pred=predictions,
													 testset=probabilities,
													 test_i=test_i,
													 y_true=sample_y[test_i],
			                                 		 scoring=scoring)

				maggr_r, maggr_s = self.mathematician.aggr(y_true=sample_y[test_i],
				                                 		   y_proba=probabilities,
				                                 		   scoring=scoring)

				aggr_r.update(caggr_r)
				aggr_r.update(aaggr_r)
				aggr_r.update(maggr_r)

				aggr_s.update(caggr_s)
				aggr_s.update(aaggr_s)
				aggr_s.update(maggr_s)

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


class InstanceDistributed(Simulator):
	"""Instance distributed classification learning simulator.

	Description:
		This class simulates a distributed learning using classifier agents. It divides
		the data horizontally, i. e., it divides the instances randomly between the learners.

	Also see: Simulator class documentation.
	"""

	@staticmethod
	def get_distribution(data, n, overlap=0):
		"""Vertically distributes the training data into n parts and returns a list of
		column indexes for each part

		Keyword arguments:
			data -- the training set to be splitted (Data)
			n -- number of divisions
			overlap -- if float, should be between 0.0 and 1.0 and represents the percentage
					   of parts' common instances. If int, should be less than or equal to the
					   number of instances and represents the number of common instances. If list,
					   represents the instances' columns indexes. By default, the value is set to 0.
		"""

		pass

	def cross_validate(self, k_fold=10, n_it=10):
		"""Runs the cross_validate function for each agent and returns a list with each learner's scores

		Keyword arguments:
			k_fold -- number of folds
			n_it -- number of cross-validation iterations (default 10, i. e., 10 k-fold cross-validation)
		"""

		pass
