import numpy as np

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

		return predi, proba


class Aggregator():
    """Aggregate classifiers predictions."""

    def __init__(self, methods=[]):
        """Set properties.

        Keyword arguments:
            methods -- a list
        """
        self.methods = methods

    def aggr(self, **kwargs):
        pass


class Voter(Aggregator):
    """Aggregate classifiers predictions by voting."""

    def __init__(self, methods=[]):
        """Set properties.

        Keyword arguments:
            methods -- social choice function's methods
        """
        self.methods = methods

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            y_proba -- learners' probabilities
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

        n_learners = len(y_proba)        # # of learners = length of proba
        _, n_classes = y_proba[0].shape  # # of classes = # of columns

        for c in range(n_classes):
            # Get class c's probabilities
            proba = [y_proba[i][:, c] for i in range(n_learners)]

            # Aggregate ranks
            sc_ranks = Profile.aggr_rank(proba, self.methods, y_pred)

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


class Combiner(Aggregator):
    """Aggregate classifiers predictions by training another classifier (combiner)."""

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            x -- combiner input
            y -- target of x
            testset -- test instances
            y_true -- true classes (for score)
            scoring -- a dict of scorers (default {})
        """
        # Get params
        x = kwargs['x']
        y = kwargs['y']
        testset = kwargs['testset']
        y_true = kwargs['y_true']
        scoring = kwargs.get('scoring', {})

        # Prep X
        n = len(x)
        X = x[0]

        for i in range(1, n):
            X = np.append(X, x[i], axis=1)

        # Prep testset
        n = len(testset)
        test = testset[0]

        for i in range(1, n):
            test = np.append(test, testset[i], axis=1)

        n = len(self.methods)
        predictions = dict()
        scores = dict()

        # For each combiner...
        for i in range(n):
            self.methods[i].fit(X, y)
            y_pred = self.methods[i].predict(test)

            k = 'cmb_' + str(i)

            predictions[k] = y_pred
            scores[k] = score(y_true, y_pred, scoring)

        return predictions, scores


class Arbiter(Aggregator):

    def __init__(self, selection_rules, methods=[]):
        """Sets the selection rule to be used.

        Keyword arguments:
            selection_rules -- a list of SelectionRule objects
            methods -- a list of classifiers (default [])
        """
        self.selection_rules = selection_rules
        super().__init__(methods)

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            learners -- a list of Learners
            x -- learners' probabilities to be trained by the arbiter
            y -- labels for x
            y_true -- true classes
            y_pred -- predicted classes
            testset -- test instances
            test_i -- test instance indexes
            scoring -- a dict of scorers (default {})
        """
        # Get params
        learners = kwargs['learners']
        y_proba = kwargs['x']
        y_train = kwargs['y']
        y_true = kwargs['y_true']
        base_pred = kwargs['y_pred']
        testset = kwargs['testset']
        scoring = kwargs.get('scoring', {})

        # Prep testset
        n = len(testset)
        test = testset[0]

        for i in range(1, n):
            test = np.append(test, testset[i], axis=1)

        n = len(self.methods)
        n_rules = len(self.selection_rules)
        n_classes = len(set(y_true))
        n_learners = len(learners)

        predictions = dict()
        scores = dict()

        for j in range(n_rules):
            t = self.selection_rules[j].select(base_pred, y_true)
            n_t = len(t)

            if n_t < 3:
                x_indices = t[0].union(t[1]) if n_t == 2 else t[0]
                x_indices = list(x_indices)

                n_instances = len(x_indices)

                x = np.zeros((n_instances, n_classes))

                for i in range(n_learners):
                    x = np.append(x, y_proba[i][x_indices, :], axis=1)

                x = x[:, n_classes:]
                y = y_train[x_indices]
            else:
                x = []
                y = []

                for i in range(n_t):
                    x_indices = list(t[i])
                    n_instances = len(x_indices)

                    xi = np.zeros((n_instances, n_classes))
                    for j in range(n_learners):
                        xi = np.append(xi, y_proba[j][x_indices, :], axis=1)

                    x.append(xi[:, n_classes:])
                    y.append(y_train[x_indices])

            # For each method...
            for i in range(n):

                if n_t < 3:
                    self.methods[i].fit(x, y)
                    y_pred = self.methods[i].predict(test)
                else:
                    y_pred = []
                    for j in range(n_t):
                        self.methods[i].fit(x[j], y[j])
                        y_pred.append(self.methods[i].predict(test))


                k = 'arb_' + str(self.selection_rules[j]) + '_' + str(i)

                predictions[k] = self.selection_rules[j].apply(base_pred, y_pred)
                scores[k] = score(y_true, predictions[k], scoring)

        return predictions, scores


class Mathematician(Aggregator):
    """Aggregate classifiers prediction by average."""

    def aggr(self, **kwargs):
        """Aggregate probabilities and return aggregated ranks and scores.

        Keyword arguments:
            y_proba -- learners' probabilities
            y_true -- true classes
            scoring -- a dict of scorers (default {})
        """
        # Get params
        y_proba = kwargs['y_proba']
        y_true = kwargs['y_true']
        scoring = kwargs.get('scoring', {})

        predictions = dict()
        results = dict()
        scores = dict()

        n_learners = len(y_proba)        # # of learners = length of proba
        _, n_classes = y_proba[0].shape  # # of classes = # of columns

        methods = self.methods.items()

        for c in range(n_classes):
            # Get class c's probabilities
            proba = [y_proba[i][:, c] for i in range(n_learners)]
            proba = np.array(proba)

            for _, operations in methods:

                for op in operations:
                    result = eval('np.{}(proba, axis=0)'.format(op))

                    results.setdefault(op, [])
                    results[op].append(result)

        for aux, operations in methods:

            for op in operations:
                results[op] = np.array(results[op])
                predictions[op] = eval('results[op].arg{}(axis=0)'.format(aux))
                scores[op] = score(y_true, predictions[op], scoring)

        return predictions, scores
