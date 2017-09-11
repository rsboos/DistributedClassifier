import numpy
import metrics

from sklearn import model_selection


class Learner():
	"""Trains a model given a classifier

	Properties:
		dataset -- A Dataset for training and testing the model (Dataset)
		classifier -- Algorithm used for training the model (an instance from sklearn library*)
		__fit -- A flag to check if the data was fit

	*The classifier should implement fit(), predict() and predict_proba().
	See the sklearn documentation for more information... 
	"""

	def __init__(self, dataset, classifier):
		"""Sets the properties, trains the data and predicts the testset
		
		Keyword arguments:
			dataset -- The Dataset used to fit a model (Dataset)
			classifier -- An instance of a classifier from sklearn library
		"""

		self.dataset = dataset
		self.classifier = classifier

	@property
	def predictions(self):
		"""Returns a predictions's ndarray copy"""
		return numpy.array(self.__predictions)

	@property
	def proba_predictions(self):
		"""Returns a proba_predictions's ndarray copy"""
		return numpy.array(self.__proba_predictions)

	def fit(self):
		"""Fits the model using the class dataset and classifier"""
		self.classifier = self.classifier.fit(self.dataset.trainingset.data, self.dataset.trainingset.target)

	def predict(self):
		"""Predict classes for the testset on dataset and returns a ndarray as result. 
		fit() is called before predict, if it was never executed."""

		return self.classifier.predict(self.dataset.testset.data)

	def predict_proba(self):
		"""Predict the probabilities for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed."""

		return self.classifier.predict_proba(self.dataset.testset.data)

	def cross_validate(self, folds, n_it=10):
		"""Computes a n_it len(folds)-fold cross-validation and returns a list of dicts of float arrays*

		Keyword arguments:
			folds -- train and test splits to cross-validate
			n_it -- number of cross-validation iterations (default 10, i. e., 10 k-fold cross-validation)

		*For more information about the returned data: 
		http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
		"""

		# Initialize an empty list for the scores
		scores = list()

		# Begins computing the n_it k-fold cross-validation
		for i in range(n_it):
			# Computes one cross-validation
			score = model_selection.cross_validate(self.classifier, self.dataset.trainingset.x, self.dataset.trainingset.y, cv=folds)
			
			# Saves the score of one iteration
			scores.append(score)

		# Returns the scores of each validation
		return scores