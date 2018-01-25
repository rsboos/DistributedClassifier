import numpy

from .metrics import score
from .data import Data, Dataset
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

	def run_fold(self, train_i, test_i, scoring={}):
		"""Generate cross-validated for an input data point.

		Keyword arguments:
			train_i -- train instances' index
			test_i -- test instances' index
			scoring -- a dict of scorers {<'scorer_name'>: <scorer_callable>}

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

		# Get scores
		scores = score(test_y, predi, scoring)

		# Transpose of proba to divide probabilities by class
		return predi, proba.T, scores

	def __choose_data(self, data):
		"""Choose the data to be used in prediction.
		If data is provided by the user, use it.
		Otherwise, the testset in dataset property.
		"""
		return self.dataset.testset.x if data is None else data


class Voter():
    """Aggregate classifiers predictions by voting."""

    def __init__(self, methods):
        """Set properties.

        Keyword arguments:
            methods -- social choice function's methods
        """
        self.methods = methods

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            proba -- dict of probabilities split in classes
            y_true -- true classes
            y_pred -- predicted classes
            scoring -- a dict of scorers (default {})
        """
        # Get params
        proba = kwargs['proba']
        y_true = kwargs['y_true']
        y_pred = kwargs['y_pred']
        scoring = kwargs.get('scoring', {})

        # rankings = a rank by class by social choice function
        class_ranks = dict()
        scores = dict()
        ranks = dict()

        # k = class' index
        for k in proba:
            sc_ranks = Profile.aggr_rank(proba[k], self.methods, y_pred)

            # Join ranks by social choice function
            for scf, r in sc_ranks.items():
                class_ranks.setdefault(scf, [])
                class_ranks[scf].append(r)

        # Get winners
        # k = social choice function
        for k in class_ranks:
            winners = join_ranks(class_ranks[k])
            metrics = score(y_true, winners, scoring)

            ranks[k] = winners   # save ranks
            scores[k] = metrics  # save scores

        return ranks, scores


class Combiner():
    """Aggregate classifiers predictions by training another classifier (combiner)."""

    def __init__(self, classifier):
        """Set properties.

        Keyword arguments:
            classifier -- classifier to be trained with classes
        """
        self.classifier = classifier

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            y_true -- true classes
            y_pred -- predicted classes
            scoring -- a dict of scorers (default {})
        """
        # Get params
        y_true = kwargs['y_true']
        y_pred = kwargs['y_pred']
        scoring = kwargs.get('scoring', {})

        # Scores and ranks by classifier
        scores = dict()
        ranks = dict()

        # Transpose y_pred to get one-instance predictions in line
        y_pred = numpy.array(y_pred)
        y_pred = y_pred.T

        # Create a learner
        data = Data(y_pred, y_true)
        dataset = Dataset(data)
        learner = Learner(dataset, self.classifier)

        # Train learner
        learner.fit()

        # Predict
        # PREDICT WITH WHICH DATA??????
        predictions = None

        ranks['combiner'] = predictions
        scores['combiner'] = score(y_true, predictions, scoring)

        return ranks, scores


class Arbiter():
	pass
