import copper
# import numpy as np
from sklearn import cross_validation


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

    def fit(self):
        for algorithm in self.algorithms:
            self.algorithms[algorithm].fit(self.X_train, self.y_train)

    def __getitem__(self, name):
        return self.algorithms[name]

    def __setitem__(self, name, value):
        self.algorithms[name] = value

    def __delitem__(self, name):
        del self.algorithms[name]

    def __len__(self):
        return len(self.algorithms)

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
