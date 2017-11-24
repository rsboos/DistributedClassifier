import numpy

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
		"""Sets the properties and fits the data

		Keyword arguments:
			dataset -- The Dataset used to fit a model (Dataset)
			classifier -- An instance of a classifier from sklearn library
		"""

		# Sets the properties
		self.dataset = dataset
		self.classifier = classifier

		# Fits the data
		self.fit()

	def fit(self):
		"""Fits the model using the class dataset and classifier"""
		self.classifier = self.classifier.fit(self.dataset.trainingset.x, self.dataset.trainingset.y)

	def predict(self, data=None):
		"""Predict classes for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed.

		Keyword arguments:
			data -- data to be predicted. When (default None), testset is used.
		"""

		testdata = self.__choose_data(data)  	  # gets the data to be predicted
		return self.classifier.predict(testdata)  # returns the predictions

	def predict_score(self, scoring, x=None, y=None):
		"""Calculate predictions' metrics and return a dict with metrics.

		Keyword arguments:
			scoring -- a dict as {<score name>: <scorer func>}
			x -- data to be predicted. When (default None), testset is used.
			y -- true pedictions. When (default None), testset is used.
		"""
		# Get predictions
		y_true = self.dataset.testset.y if y is None else y
		y_pred = self.predict(x)

		# Init dict
		scores = dict()

		# For each metric...
		for k in scoring:
			# Get the scorer function
			scorer = scoring[k]

			# Calculate and save score
			scores['test_' + k] = scorer._score_func(y_true, y_pred)

		return scores

	def predict_proba(self, data=None):
		"""Predict the probabilities for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed.

		Keyword arguments:
			data -- data to be predicted. When (default None), testset is used.
		"""

		testdata = self.__choose_data(data)  			# gets the data to be predicted
		return self.classifier.predict_proba(testdata)  # returns the predictions

	def cross_validate(self, folds, scoring=['accuracy', 'precision']):
		"""Computes a len(folds)-fold cross-validation and returns a list of dicts of float arrays*

		Keyword arguments:
			folds -- train and test splits to cross-validate*
			scoring -- metrics to be returned (default ['accuracy', 'precision'])*

		*For more information about the returned data and the parameters:
		http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
		"""
		# Short names
		x = self.dataset.trainingset.x # instances
		y = self.dataset.trainingset.y # classes

		# Compute cross-validation and return its score
		return model_selection.cross_validate(self.classifier, x, y, cv=folds, scoring=scoring)

	def __choose_data(self, data):
		"""Choose the data to be used in prediction.
		If data is provided by the user, use it.
		Otherwise, the testset in dataset property.
		"""

		# If data is not provided...
		if data is None:
			return self.dataset.testset.x

		# Data was provided
		return data
