import copper
from sklearn import cross_validation

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss


class ModelComparison(dict):
    """ Utility for easy model(algorithm) comparison.
    Can use only numpy arrays or copper.Dataset to generate the training and
    testing datasets.

    Note: Designed to work with sklearn algorithms (extending BaseEstimator)
    but not necesary if the algorithm matches the basic sklearn API:
    algo.fit, algo.predict, algo.predict_proba

    Parameters
    ----------

    Examples
    --------
    >>> train = copper.Dataset(...)
    >>> test = copper.Dataset(...)
    >>> mc = copper.ModelComparison(...)
    >>> from sklearn.linear_model import LogisticRegression
    >>> mc['LR'] = LogisticRegression()
    >>> mc['LR with p=l1'] = LogisticRegression(penalty='l1')
    >>> mc.fit()
    >>> mc.predict()
    np.array([[...]])
    """

    def __init__(self):
        self.algorithms = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.le = None

    def get_train(self):
        return self.X_train, self.y_train

    def set_train(self, dataset):
        assert type(dataset) is copper.Dataset, "Should be a copper.Dataset"
        self.X_train = copper.t.ml_inputs(dataset)
        self.le, self.y_train = copper.t.ml_target(dataset)

    def get_test(self):
        return self.X_test, self.y_test

    def set_test(self, dataset):
        assert type(dataset) is copper.Dataset, "Should be a copper.Dataset"
        self.X_test = copper.t.ml_inputs(dataset)
        _, self.y_test = copper.t.ml_target(dataset)

    train = property(get_train, set_train, None)
    test = property(get_test, set_test, None)

    def train_test_split(self, dataset, **args):
        """ Random split of a copper.Datasetinto a training and testing
        datasets

        Arguments are the same as: sklearn.cross_validation.train_test_split
        only test_size is necessary.

        Parameters
        ----------
        test_size: float percentage of the dataset used for testing
            between 0 and 1.
        """
        assert type(dataset) is copper.Dataset, "Should be a copper.Dataset"
        inputs = copper.t.ml_inputs(dataset)
        self.le, target = copper.t.ml_target(dataset)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            cross_validation.train_test_split(inputs, target, **args)

    def __getitem__(self, name):
        return self.algorithms[name]

    def __setitem__(self, name, value):
        self.algorithms[name] = value

    def __delitem__(self, name):
        del self.algorithms[name]

    def __len__(self):
        return len(self.algorithms)

    def parse_entries(self, X_test=None, y_test=None):
        """ DRY: Small utility used inside of the class.
        """
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        elif isinstance(X_test, copper.Dataset):
            X_test = copper.transforms.ml_inputs(X_test)
            _, y_test = copper.transforms.ml_target(X_test)
        assert X_test is not None, 'Nothing to predict'
        return X_test, y_test

# --------------------------- SKLEARN API -------------------------------------

    def fit(self):
        for algorithm in self.algorithms:
            self.algorithms[algorithm].fit(self.X_train, self.y_train)

    def predict(self, X_test=None):
        X_test, _ = self.parse_entries(X_test, None)

        ans = pd.DataFrame(index=range(len(X_test)))
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            scores = algo.predict(X_test)
            new = pd.Series(scores, index=ans.index, name=alg_name, dtype=int)
            ans = ans.join(new)
        return ans

    def predict_proba(self, X_test=None):
        X_test, _ = self.parse_entries(X_test, None)

        ans = pd.DataFrame(index=range(len(X_test)))
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            probas = algo.predict_proba(X_test)
            for val in range(probas.shape[1]):
                new = pd.Series(probas[:, val], index=ans.index)
                new.name = '%s [%d]' % (alg_name, val)
                ans = ans.join(new)
        return ans

# ------------------------- SKLEARN METRICS -----------------------------------

    def metric(self, func, X_test=None, y_test=None, name='', ascending=False, **args):
        X_test, y_test = self.parse_entries(X_test, y_test)

        ans_index = []
        ans_value = []
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            y_pred = algo.predict(X_test)
            scores = func(y_test, y_pred, **args)

            if isinstance(scores, np.ndarray):
                for i, score in enumerate(scores):
                    ans_index.append('%s (%i)' % (alg_name, i))  # Change i for label
                    ans_value.append(score)
            else:
                ans_index.append(alg_name)
                ans_value.append(scores)
        return pd.Series(ans_value, index=ans_index).order(ascending=ascending)

    def accuracy_score(self, **args):
        return self.metric(accuracy_score, name='Accuracy', **args)

    def auc_score(self, **args):
        return self.metric(auc_score, name='AUC', **args)

    def average_precision_score(self, **args):
        return self.metric(average_precision_score, name='Avg Precision', **args)

    def f1_score(self, **args):
        return self.metric(f1_score, name='F1', **args)

    def fbeta_score(self, beta=1, **args):
        return self.metric(fbeta_score, name='Fbeta', beta=beta, **args)

    def hinge_loss(self, **args):
        return self.metric(hinge_loss, name='Hinge loss', **args)

    def matthews_corrcoef(self, **args):
        return self.metric(matthews_corrcoef, name='Matthews Coef', **args)

    def precision_score(self, **args):
        return self.metric(precision_score, name='Precision', **args)

    def recall_score(self, **args):
        return self.metric(recall_score, name='Recall', **args)

    def zero_one_loss(self, **args):
        return self.metric(zero_one_loss, name='Zero one loss', ascending=True, **args)

# ------------------------- CONFUSION MATRIX ----------------------------------

    def _cm(self, X_test=None, y_test=None):
        '''
        Calculates the confusion matrixes of the classifiers

        Parameters
        ----------
            clfs: list or str, of the classifiers to calculate the cm

        Returns
        -------
            python dictionary
        '''
        X_test, y_test = self.parse_entries(X_test, y_test)

        ans = {}
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            y_pred = algo.predict(self.X_test)
            ans[alg_name] = confusion_matrix(y_test, y_pred)
        return ans

    def cm(self, clf, X_test=None, y_test=None):
        '''
        Return a pandas.DataFrame version of a confusion matrix

        Parameters
        ----------
            clf: str, classifier identifier
        '''
        cm = self._cm(X_test, y_test)[clf]
        values = np.unique(self.y_test)
        return pd.DataFrame(cm, index=values, columns=values)

#           ---------    TESTS
import numpy as np
import pandas as pd
from nose.tools import raises, ok_
from copper.tests.utils import eq_


def get_iris():
    from sklearn import datasets
    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target
    return X, Y

def get_iris_ds():
    X, Y = get_iris()
    df = pd.DataFrame(X)
    df['Target'] = pd.Series(Y, name='Target')

    ds = copper.Dataset(df)
    ds.role['Target'] = ds.TARGET
    return ds

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
