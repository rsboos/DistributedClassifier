import numpy

from .metrics import score
from .data import Data
from sklearn import model_selection
from .metrics import score, join_ranks
from social_choice.profile import Profile


class Learner():
	"""Trains a model given a classifier

	Properties:
		dataset -- A Data for training and testing the model
		classifier -- Algorithm used for training the model (an instance from sklearn library*)
		__fit -- A flag to check if the data was fit

	*The classifier should implement fit(), predict() and predict_proba().
	See the sklearn documentation for more information...
	"""

	def __init__(self, dataset, classifier):
		"""Sets the properties and fits the data

		Keyword arguments:
			dataset -- The Data used to fit a model
			classifier -- An instance of a classifier from sklearn library
		"""

		# Sets the properties
		self.dataset = dataset
		self.classifier = classifier

	def fit(self):
		"""Fits the model using the class dataset and classifier"""
		self.classifier = self.classifier.fit(self.dataset.x, self.dataset.y)

	def predict(self, data):
		"""Predict classes for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed.

		Keyword arguments:
			data -- data to be predicted
		"""
		return self.classifier.predict(data)  # returns the predictions

	def predict_proba(self, data):
		"""Predict the probabilities for the testset on dataset and returns a ndarray as result.
		fit() is called before predict, if it was never executed.

		Keyword arguments:
			data -- data to be predicted
		"""
		return self.classifier.predict_proba(data)  # returns the predictions

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
		train_x = self.dataset.x[train_i, :]
		train_y = self.dataset.y[train_i]

		# Get test data
		test_x = self.dataset.x[test_i, :]
		test_y = self.dataset.y[test_i]

		# Train model
		self.classifier.fit(train_x, train_y)

		# Get model's predictions
		predi = self.classifier.predict(test_x)
		proba = self.classifier.predict_proba(test_x)

		# Get scores
		scores = score(test_y, predi, scoring)

		# Transpose of proba to divide probabilities by class
		return predi, proba, scores


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
            y_proba -- dict of probabilities split in classes
            y_true -- true classes
            y_pred -- predicted classes
            scoring -- a dict of scorers (default {})
        """
        # Get params
        y_proba = kwargs['y_proba']
        y_true = kwargs['y_true']
        y_pred = kwargs['y_pred']
        scoring = kwargs.get('scoring', {})

        # rankings = a rank by class by social choice function
        class_ranks = dict()
        scores = dict()
        ranks = dict()

        n_learners = len(y_proba)       # # of learners = length of proba
        n_classes = len(y_proba[0])     # # of classes =

        # for i in range(n_learners):

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

    def __init__(self, methods):
        """Set properties.

        Keyword arguments:
            methods -- a list of classifiers
        """
        self.methods = methods

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            y_proba -- dict of probabilities split in classes
            y_true -- true classes
            y_pred -- predicted classes
            scoring -- a dict of scorers (default {})
        """
        # Get params
        y_proba = kwargs['y_proba']
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


class Arbiter():
	pass
