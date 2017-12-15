import numpy
import math
import copy

from .agents import Learner
from .data import Data, Dataset
from sklearn import model_selection
from .metrics import cv_score, score, join_ranks
from social_choice.profile import Profile


class Simulator():
	"""Distributed classification learning simulator.

	Properties:
		learners -- a list of learners
	"""

	def __init__(self, learners):
		"""Sets the properties, create the agents, divides the data between the agents

		Keyword arguments:
			n_learners -- number of agents to be used (must be greater than 1)
		"""

		self.learners = learners

	def fit(self):
		"""Train model with trainingset for each learner."""
		[learner.fit() for learner in self.learners]

	def predict_proba(self, data=None):
		"""Predicts the probabilities using the learner classifier
		and returns a list of predictions for every learner

		Keyword arguments:
			data -- data to be predicted. When (default None), testeset is used.
		"""
		return [learner.predict_proba(data).T for learner in self.learners]

	@staticmethod
	def get_distribution(data, n, overlap=0):
		"""Not implemented. Should be implemented in a child class."""
		pass

	@classmethod
	def load(cls, data, classifers, overlap=0, test_size=0.3):
		"""Not implemented. Should be implemented in a child class."""
		pass

	def cross_validate(self, k_fold=10, n_it=10):
		"""Not implemented. Should be implemented in a child class."""
		pass


class FeatureDistributed(Simulator):
	"""Feature distributed classification learning simulator.

	Description:
		This class simulates a distributed learning using classifier agents. It divides
		the data vertically, i. e., it divides the features randomly between the learners.

	Also see: Simulator class documentation.
	"""

	def __init__(self, learners):
		"""Sets the properties, create the agents, divides the data between the agents

		Keyword arguments:
			learners -- list of learners (lenght must be greater than 1)
		"""

		# Calls __init__ from parent
		super().__init__(learners)


	@staticmethod
	def get_distribution(data, n, overlap=0):
		"""Vertically distributes the training data into n parts and returns a list of
		column indexes for each part

		Keyword arguments:
			data -- the training set to be splitted (Data)
			n -- number of divisions
			overlap -- if float, should be between 0.0 and 1.0 and represents the percentage
					   of parts' common features. If int, should be less than or equal to the
					   number of features and represents the number of common features. If list,
					   represents the features' columns indexes. By default, the value is set to 0.
		"""

		# Check the type of overlap variable and sets the n_common and common_features, where
		# n_common is the number of common features for each part and
		# common_features is a common features' index's list
		if type(overlap) is int:
			common_features = set(numpy.random.choice(data.n_features, overlap))  # gets a n length list with random numbers between 0-n_features

		elif type(overlap) is float:
			n_common = math.ceil(data.n_features * overlap)						  # calculates the amount of features each part has in common
			common_features = set(numpy.random.choice(data.n_features, n_common)) # gets a n length list with random numbers between 0-n_features

		elif type(overlap) is list:
			common_features = set(overlap)

		else: # is neither int, float and list
			raise TypeError("%s should be a float, an int or a list. %s given." % (str(overlap), type(overlap)))

		# Calculates the distinct features' indexes
		distinct_features = list(set(range(data.n_features)) - common_features) # common_features' complement

		# Just converts from set to list
		common_features = list(common_features)

		# Permutates the distinct features (randomize)
		distinct_features = list(numpy.random.permutation(distinct_features))

		# Calculate the number of distinct features per part
		n_distinct = len(distinct_features)

		# Calculates the number of distinc features per part
		n_part_features = math.ceil(n_distinct / n)

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

	@classmethod
	def load(cls, data, classifiers, overlap=0, test_size=0.3):
		"""Creates n_learners Learner objects and returns a DistributedClassification object

		Keyword arguments:
			data -- a Data object
			classifiers -- a list of classifiers, len(classifiers) define the number of learners
			overlap -- if float, should be between 0.0 and 1.0 and represents the percentage
					   of parts' in common. If int, should be less than or equal to the
					   number of features/instances and represents the number of common
					   features/instances. If list, represents the features'/instances'
					   indexes. By default, the value is set to 0.
			test_size -- percent of test instances (default 0.3)
		"""

		# The number of classifiers define the number of learners
		n_learners = len(classifiers)

		# Train and test split
		# This guarantees that every learner has the same splits
		train, test = Dataset.train_test_split(data, test_size)

		# Gets the indexes to slice the data
		# Because the distribution is vertical, we can use a part
		# of the data with all features to get the slice indexes
		indexes = cls.get_distribution(train, n_learners, overlap)

		# Convert to ndarray for slicing
		train.x, test.x = numpy.array(train.x), numpy.array(test.x)
		train.y, test.y = numpy.array(train.y), numpy.array(test.y)

		# Initialize an empty list for learners
		learners = list()

		# For each learner
		for i in range(n_learners):
			features = indexes[i]								 # gets the features indexes

			trainingset = Data(train.x[:, features], train.y)	 # gets the trainingset
			testset = Data(test.x[:, features], test.y)		     # gets the testset

			dataset = Dataset(trainingset, testset) 	     	 # creates a dataset for learner
			classifier = classifiers[i]			   			     # gets the classifier

			learner = Learner(dataset, classifier) 			     # creates the learner
			learners.append(learner)			   			     # saves the learner

		# Creates a DistributedClassifition simulator
		return cls(learners)

	def predict(self, scf, scoring={}, data=None):
		"""Predicts using the learners' classifiers.

		Keyword arguments:
			scf -- social choice functions as list of strings
			scoring -- a dict of scorers (default {})
			data -- data to be predicted. When (default None), testeset is used.
		"""
		# Initialize empty list for probabilities
		classif_scores = dict()
		probabilities = dict()
		predictions = list()

		# For each learner...
		for i in range(len(self.learners)):
			y_pred = self.learners[i].predict(data)			# predicted classes
			proba = self.learners[i].predict_proba(data)     # probabilities of each class
			proba = proba.T 								# split by class

			# Save predictions
			predictions.append(y_pred)

			# Get and save score
			y_true = self.learners[i].dataset.testset.y
			metrics = score(y_true, y_pred, scoring)
			classif_scores[i] = metrics

			# Save proba by k-class rank
			for k in range(len(proba)):
				probabilities.setdefault(k, [])
				probabilities[k].append(proba[k])

		# Aggregate probabilities
		rank, rank_score = self.aggr_probabilities(probabilities, scf, y_true, predictions, scoring)

		return rank, classif_scores, rank_score

	def aggr_probabilities(self, proba, sc_functions, y_true, y_pred, scoring={}):
		"""Aggregate probabilities and return aggregated ranks and scores.

		Keyword arguments:
			proba -- dict of probabilities split in classes
			sc_function -- list of social choice functions name
			y_true -- true classes
			y_pred -- predicted classes
			scoring -- a dict of scorers (default {})
		"""
		# rankings = a rank by class by social choice function
		class_ranks = dict()
		scores = dict()
		ranks = dict()

		# k = class' index
		for k in proba:
			sc_ranks = Profile.aggr_rank(proba[k], sc_functions, y_pred)

			# Join ranks by social choice function
			for scf, r in sc_ranks.items():
				class_ranks.setdefault(scf, [])
				class_ranks[scf].append(r)

		# Get winners
		# k = social choice function
		for k in class_ranks:
			winners = join_ranks(class_ranks[k])
			metrics = score(y_true, winners, scoring)

			ranks[k] = winners	 # save ranks
			scores[k] = metrics  # save scores

		return ranks, scores

	def cross_validate(self, sc_functions, k_fold=10, scoring={}, n_it=10):
		"""Runs the cross_validate function for each agent and returns a list with each learner's scores

		Keyword arguments:
			sc_functions -- a list with social choice functions' names
			k_fold -- number of folds (default 10)
			scoring -- metrics to be returned (default {})*
			n_it -- number of cross-validation iterations (default 10, i. e., 10 k-fold cross-validation)

		*For more information about the returned data and the parameters:
		http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

		For how to use scoring:
		http://scikit-learn.org/stable/modules/cross_validation.html
		"""
		# Number of learners
		n = len(self.learners)

		# Initializes an empty dict for scores and rankings
		classif_scores = dict()
		rank_scores = dict()
		ranks = dict()

		# Gets a sample of the data for the splitter
		sample_x = self.learners[0].dataset.trainingset.x # instances
		sample_y = self.learners[0].dataset.trainingset.y # classes

		for i in range(n_it):
			# Splits into k training and test folds for cross-validation
			skf = model_selection.StratifiedKFold(n_splits=k_fold) # create the 'splitter' object

			# Create folds and iterate
			for train_i, test_i in skf.split(sample_x, sample_y):
				# Initialize empty list for probabilities
				probabilities = dict()
				predictions = list()

				# For each learner...
				for j in range(n):
					# Compute a fold
					pred, proba, metrics = self.learners[j].run_fold(train_i, test_i, scoring)

					# Save predictions
					predictions.append(pred)

					# Save proba by k-class rank
					for k in range(len(proba)):
						probabilities.setdefault(k, [])
						probabilities[k].append(proba[k])

					# Save score
					classif_scores.setdefault(j, [])
					classif_scores[j].append(metrics)

				# Get true test classes
				y_true = sample_y[test_i]

				# Aggregate probabilities
				aggr_r, aggr_s = self.aggr_probabilities(probabilities, sc_functions, y_true, predictions, scoring)

				# Save ranks
				for k in aggr_r:
					ranks.setdefault(k, [])
					ranks[k].append(aggr_r[k])

				# Save scores
				for k in aggr_s:
					rank_scores.setdefault(k, [])
					rank_scores[k].append(aggr_s[k])

		# Return the ranks and aggregated scores as DataFrames for each learner
		return ranks, [cv_score(classif_scores[k]) for k in classif_scores], [cv_score(rank_scores[k]) for k in rank_scores]				# ranks' scores


class InstanceDistributed(Simulator):
	"""Instance distributed classification learning simulator.

	Description:
		This class simulates a distributed learning using classifier agents. It divides
		the data horizontally, i. e., it divides the instances randomly between the learners.

	Also see: Simulator class documentation.
	"""

	def __init__(self, learners):
		"""Sets the properties, create the agents, divides the data between the agents

		Keyword arguments:
			learners -- list of learners (lenght must be greater than 1)
		"""

		# Calls __init__ from parent
		super().__init__(learners)

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
