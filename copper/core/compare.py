import copper
# import numpy as np
from sklearn import cross_validation

class ModelComparison(dict):

    def __init__(self):
        self.algorithms = {}
        self.X_train = None
        self.y_train = None
        self.X_test  = None
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

    def sample(self, dataset, **args):
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
