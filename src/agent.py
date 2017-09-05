class Agent():
	"""Trains a model given a classifier

	Properties:
		dataset -- A Dataset for training and testing the model (Dataset)
		classifier -- Algorithm used for training the model (an instance from sklearn library*)
		__fit -- A flag to check if the data was fit

	*The classifier should implement fit(), predict() and predict_proba().
	See the sklearn documentation for more information... 
	"""

	def __init__(self, dataset, classifier):
		"""Sets the properties
		
		Keyword arguments:
			dataset -- The Dataset used to fit a model (Dataset)
			classifier -- An instance of a classifier from sklearn library
		"""

		self.dataset = dataset
		self.classifier = classifier
		self.__fit = False

	def fit(self):
		"""Fits the model using the class dataset and clissifier"""

		self.classifier = self.classifier.fit(self.dataset.trainingset.data, self.dataset.trainingset.target)
		self.__fit = True

	def predict(self):
		"""Predict classes for the testset on dataset and returns a ndarray as result. 
		fit() is called before predict, if it was never executed.
		"""

		self.__check_fit()
		return self.classifier.predict(self.dataset.testset.data)

	def predict_proba(self):
		"""Predict the probabilities for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed.
		"""

		self.__check_fit()
		return self.classifier.predict_proba(self.dataset.testset.data)

	def __check_fit(self):
		"""Check if fit() was executed. If not, executes fit()."""
		
		if not self.__fit:
			self.fit()