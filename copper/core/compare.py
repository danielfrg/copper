import copper
# import numpy as np
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


class ModelComparison(dict):
    """ Utility for easy model(algorithm) comparison.
    Can use only numpy arrays or copper.Dataset to generate the training and
    testing datasets.

    Note: Designed to work with sklearn algorithms (extending BaseEstimator)
    but not necesary if the algorithm matches the basic skelarn API:
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

    def get_train(self):
        return self.X_train, self.y_train

    def set_train(self, dataset):
        assert type(dataset) is copper.Dataset, "Should be a copper.Dataset"
        self.X_train = copper.utils.transforms.ml_inputs(dataset)
        self.y_train = copper.utils.transforms.ml_target(dataset)

    def get_test(self):
        return self.X_test, self.y_test

    def set_test(self, dataset):
        assert type(dataset) is copper.Dataset, "Should be a copper.Dataset"
        self.X_test = copper.utils.transforms.ml_inputs(dataset)
        self.y_test = copper.utils.transforms.ml_target(dataset)

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
        inputs = copper.utils.transforms.ml_inputs(dataset)
        target = copper.utils.transforms.ml_target(dataset)
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

# --------------------------- SKLEARN API -------------------------------------

    def fit(self):
        for algorithm in self.algorithms:
            self.algorithms[algorithm].fit(self.X_train, self.y_train)

    def predict(self, X_test=None):
        if X_test is None:
            X_test = self.X_test
        elif isinstance(X_test, copper.Dataset):
            X_test = copper.transforms.ml_inputs(X_test)
        assert X_test is not None, 'Nothing to predict'

        ans = pd.DataFrame(index=range(len(X_test)))
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            scores = algo.predict(X_test)
            new = pd.Series(scores, index=ans.index, name=alg_name, dtype=int)
            ans = ans.join(new)
        return ans

    def predict_proba(self, X_test=None):
        if X_test is None:
            X_test = self.X_test
        elif isinstance(X_test, copper.Dataset):
            X_test = copper.transforms.ml_inputs(X_test)
        assert X_test is not None, 'Nothing to predict'

        ans = pd.DataFrame(index=range(len(X_test)))
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            probas = algo.predict_proba(X_test)
            for val in range(probas.shape[1]):
                new = pd.Series(probas[:, val], index=ans.index)
                new.name = '%s [%d]' % (alg_name, val)
                ans = ans.join(new)
        return ans


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
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        elif isinstance(X_test, copper.Dataset):
            X_test = copper.transforms.ml_inputs(X_test)
            y_test = copper.transforms.ml_target(X_test)
        assert X_test is not None, 'Nothing to predict'

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
        values = set(self.y_test)
        return pd.DataFrame(cm, index=values, columns=values)

# ----------------------------- METRICS ---------------------------------------

    def metric(self, func, X_test=None, y_test=None, name='', ascending=False):
        if X_test is None and y_test is None:
            X_test = self.X_test
            y_test = self.y_test
        elif isinstance(X_test, copper.Dataset):
            X_test = copper.transforms.ml_inputs(X_test)
            y_test = copper.transforms.ml_target(X_test)
        assert X_test is not None, 'Nothing to predict'

        ans = pd.Series(index=self.algorithms.keys(), name=name)
        for alg_name in self.algorithms:
            algo = self.algorithms[alg_name]
            y_pred = algo.predict(X_test)
            ans[alg_name] = func(y_test, y_pred)
        return ans.order(ascending=ascending)

    def accuracy(self, *args):
        from sklearn.metrics import accuracy_score
        return self.metric_wrapper(accuracy_score, name='Accuracy', *args)


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

def test_metric_wrapper():
    ds = get_iris_ds()
    mc = ModelComparison()
    mc.train_test_split(ds)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    mc['LR'] = LogisticRegression()
    mc['SVM'] = SVC(probability=True)
    mc.fit()
    print mc.accuracy()

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
