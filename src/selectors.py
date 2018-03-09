import numpy as np


class SelectionRule():

    @classmethod
    def normalize(cls, pred):
        """Sum of predicition's labels for each class.

        Arguments
            pred: a list of predicitions

        Return
            a pair of lists (list of classes, sum of each class)
        """
        elems, counts = np.unique(pred, return_counts=True)
        predset = list(zip(elems, counts))

        cc = [(i, 0) for i in range(len(predset)) if i != predset[i][0]]
        cc = sorted(cc + predset, key=lambda x: x[0])

        return zip(*cc)

    @classmethod
    def agree(cls, pred):
        """Agreement by majority.

        Argument
            pred: a list of predictions

        Return
            true or false
        """
        classes, countings = cls.normalize(pred)
        ratio = np.array(countings) / pred.size
        consensus = np.where(ratio > 0.5)

        return bool(consensus[0].size)

    @classmethod
    def correct(cls, pred, y):
        """Correctness by majority, i.e., if predictions that agree are agreeing on the right label.

        Argument
            pred: a list of predictions that agree on some label
            y: true label

        Return
            true or false
        """
        classes, countings = cls.normalize(pred)
        return np.argmax(countings) == y

    @classmethod
    def select(cls, y_pred, y_true):
        """Select instances according to some rule.

        Arguments
            y_pred: a nested list of predictions
            y_true: a list of true labels

        Return
            tuple
        """
        raise NotImplemented

    @classmethod
    def apply(cls, base_pred, arbiter_pred):
        """Apply an arbitery rule to predicitions.

        Arguments
            base_pred: a nested list of predictions
            arbiter_pred: the arbiters' predictions

        Return
            a list of predicitions
        """
        n_learners = len(base_pred)
        n_pred = len(arbiter_pred)

        predictions = []

        for j in range(n_pred):
            pred = np.array([base_pred[i][j] for i in range(n_learners)])

            if cls.agree(pred):
                _, c = cls.normalize(pred)
                predictions.append(np.argmax(c))
            else:
                predictions.append(arbiter_pred[j])

        return predictions


class MetaDiff(SelectionRule):

    @classmethod
    def select(cls, y_pred, y_true):
        """Select instances that disagree.

        Arguments
            y_pred: a nested list of predictions
            y_true: a list of true labels

        Return
            tuple (set of instances' indexes, )
        """
        n_learners = len(y_pred)
        n_pred = len(y_true)

        indices = []

        for j in range(n_pred):
            pred = np.array([y_pred[i][j] for i in range(n_learners)])

            if not cls.agree(pred):
                indices.append(j)

        return (set(indices),)

    def __str__(self):
        return 'md'


class MetaDiffInc(MetaDiff):

    @classmethod
    def select(cls, y_pred, y_true):
        """Select instances that disagree and agree but are incorrect.

        Arguments
            y_pred: a nested list of predictions
            y_true: a list of true labels

        Return
            tuple (set of instances' indexes that disagree, set of instances' indexes that agree incorrectly)
        """
        td = super().select(y_pred, y_true)

        n_learners = len(y_pred)
        n_pred = len(y_true)

        indices = []

        pred_i = list(range(n_pred))
        for j in set(pred_i).difference(td[0]):
            pred = np.array([y_pred[i][j] for i in range(n_learners)])

            if not cls.correct(pred, y_true[j]):
                indices.append(j)

        return (td[0], set(indices))

    def __str__(self):
        return 'mdi'


class MetaDiffIncCorr(MetaDiffInc):

    @classmethod
    def select(cls, y_pred, y_true):
        """Select instances that disagree and agree.

        Arguments
            y_pred: a nested list of predictions
            y_true: a list of true labels

        Return
            tuple (set of instances' indexes that disagree,
                   set of instances' indexes that agree incorrectly,
                   set of instances' indexes that agree correctly)
        """
        td, ti = super().select(y_pred, y_true)

        tdi = td.union(ti)
        n_pred = len(y_true)

        return (td, ti, set(range(n_pred)).difference(tdi))

    @classmethod
    def apply(cls, base_pred, arbiter_pred):
        """Apply an arbitery rule to predicitions.

        Arguments
            base_pred: a nested list of predictions
            arbiter_pred: the arbiters' predictions

        Return
            a list of predicitions
        """
        n_learners = len(base_pred)
        n_pred = len(arbiter_pred[0])

        ad_pred = arbiter_pred[0]
        ai_pred = arbiter_pred[1]
        ac_pred = arbiter_pred[2]

        predictions = []

        for j in range(n_pred):
            pred = np.array([base_pred[i][j] for i in range(n_learners)])

            if not cls.agree(pred):
                predictions.append(ad_pred[j])
            elif cls.agree(np.append(pred, [ac_pred[j]])): # PODEMOS COLOCAR DUAS VEZES A PREDIÇÃO DO ARBITRO PARA FAVORECÊ-LO
                predictions.append(ac_pred[j])
            else:
                predictions.append(ai_pred[j])

        return predictions

    def __str__(self):
        return 'mdic'
