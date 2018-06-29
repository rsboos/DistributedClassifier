import numpy as np

from .metrics import score, join_ranks
from social_choice.profile import Profile
from .selectors import MetaDiff, MetaDiffInc, MetaDiffIncCorr


class Learner():
	"""Train a model given a classifier.

	Properties:
		X -- a training set
        y -- a target set
        classifier -- An instance of a classifier from sklearn library*

    *The classifier should implement fit(), predict() and predict_proba().
    See the sklearn documentation for more information...
	"""
	def __init__(self, X, y, classifier):
		self.X = X
		self.y = y
		self.classifier = classifier

	def fit(self, X=None, y=None):
		"""Fit the model using the class dataset and classifier.

        Keyword arguments:
            X -- training set (default self.x)
            y -- target set (default self.y)
        """
		if X is None:
			X = self.X

		if y is None:
			y = self.y

		self.classifier.fit(X, y)

	def predict(self, X):
		"""Predict classes for the testset on dataset and returns a ndarray as result.

		Keyword arguments:
			X -- data to be predicted
		"""
		return self.classifier.predict(X)

	def predict_proba(self, X):
		"""Predict the probabilities for the testset on dataset and returns a ndarray as result.

		Keyword arguments:
			X -- data to be predicted
		"""
		return self.classifier.predict_proba(X)

	def evaluate(self, fold, scoring={}):
		"""Generate cross-validated for an input data point.

		Keyword arguments:
			folds -- CV folds for one run
            scoring -- metrics to be returned (default {})*
		"""
		train_i, val_i, test_i = fold

		x_train = self.X[train_i, :]
		y_train = self.y[train_i]

		x_val = self.X[val_i, :]
		y_val = self.y[val_i]

		x_test = self.X[test_i, :]
		y_test = self.y[test_i]

		self.fit(x_train, y_train)

		y_pred = self.predict(x_test)
		y_proba_val = self.predict_proba(x_val)
		y_proba_test = self.predict_proba(x_test)

		metrics = score(y_test, y_pred, scoring)

		return y_pred, y_proba_val, y_proba_test, metrics


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

    def __init__(self, selection_rule, methods=[]):
        """Sets the selection rule to be used.

        Keyword arguments:
            selection_rule -- SelectionRule object
            methods -- a list of classifiers (default [])
        """
        self.selection_rule = selection_rule
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
        n_classes = len(set(y_true))
        n_learners = len(learners)
        n_instances = y_proba[0].shape[0]

        # Prep trainingset
        x = np.zeros((n_instances, n_classes))

        for i in range(n_learners):
            x = np.append(x, y_proba[i], axis=1)

        x = x[:, n_classes:]

        predictions = dict()
        scores = dict()

        selection = self.selection_rule.select(base_pred, y_true)
        xt, yt = self.get_from_selection(x, y_train, selection)

        # For each method...
        for i in range(n):
            method = self.fit(xt, yt, self.methods[i], x, y_train)
            y_pred = self.predict(test, method)

            k = str(self) + '_' + str(i)

            predictions[k] = self.selection_rule.apply(base_pred, y_pred)
            scores[k] = score(y_true, predictions[k], scoring)

        return predictions, scores

    def get_from_selection(self, x, y_train, selection):
        raise NotImplemented

    def fit(self, Xs, ys, method, X, y):
        missing_classes = set(y) - set(ys)

        if len(missing_classes) == 0 and len(ys) > 4:  # if there is enough data to learn form
            return method.fit(Xs, ys)

        return method.fit(X, y)

    def predict(self, X, method):
        return method.predict(X)

    def __str__(self):
        return 'arb'


class ArbiterMetaDiff(Arbiter):

    def __init__(self, methods=[]):
        super().__init__(MetaDiff(), methods)

    def get_from_selection(self, x, y_train, selection):
        x_indices = selection[0]
        x_indices = list(x_indices)

        xt = x[x_indices, :]
        yt = y_train[x_indices]

        return xt, yt

    def __str__(self):
        return super().__str__() + '_md'


class ArbiterMetaDiffInc(Arbiter):

    def __init__(self, methods=[]):
        super().__init__(MetaDiffInc(), methods)

    def get_from_selection(self, x, y_train, selection):
        x_indices = selection[0].union(selection[1])
        x_indices = list(x_indices)

        xt = x[x_indices, :]
        yt = y_train[x_indices]

        return xt, yt

    def __str__(self):
        return super().__str__() + '_mdi'


class ArbiterMetaDiffIncCorr(Arbiter):

    def __init__(self, methods=[]):
        super().__init__(MetaDiffIncCorr(), methods)

    def get_from_selection(self, x, y_train, selection):
        n = len(selection)
        xt = []
        yt = []

        for i in range(n):
            x_indices = list(selection[i])
            xt.append(x[x_indices, :])
            yt.append(y_train[x_indices])

        return xt, yt

    def fit(self, Xs, ys, method, X, y):
        n = len(Xs)
        methods = []
        for i in range(n):
            m = super().fit(Xs[i], ys[i], method, X, y)
            methods.append(m)

        return methods

    def predict(self, X, method):
        n = len(method)
        predictions = []

        for i in range(n):
            y_pred = super().predict(X, method[i])
            predictions.append(y_pred)

        return predictions

    def __str__(self):
        return super().__str__() + '_mdic'


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
