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

	def predict_proba(self, data=None):
		"""Predict the probabilities for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed.

		Keyword arguments:
			data -- data to be predicted. When (default None), testset is used.
		"""
		testdata = self.__choose_data(data)  			# gets the data to be predicted
		return self.classifier.predict_proba(testdata)  # returns the predictions

	def run_fold(self, train_i, test_i):
		"""Generate cross-validated for an input data point.

		Keyword arguments:
			train_i -- train instances' index
			test_i -- test instances' index

		*For more information about the returned data and the parameters:
		http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
		"""
		# Get training data
		train_x = self.dataset.trainingset.x[train_i, :]
		train_y = self.dataset.trainingset.y[train_i]

		# Get test data
		test_x = self.dataset.trainingset.x[test_i, :]
		test_y = self.dataset.trainingset.y[test_i]

		# Train model
		self.classifier.fit(train_x, train_y)

		# Get model's predictions
		predi = self.classifier.predict(test_x)
		proba = self.classifier.predict_proba(test_x)

		# Transpose of proba to divide probabilities by class
		return predi, proba.T

	def __choose_data(self, data):
		"""Choose the data to be used in prediction.
		If data is provided by the user, use it.
		Otherwise, the testset in dataset property.
		"""
		return self.dataset.testset.x if data is None else data
