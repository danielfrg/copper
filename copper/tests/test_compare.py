from __future__ import division
import copper
import numpy as np
import pandas as pd
from nose.tools import raises, ok_
from copper.tests.utils import eq_
from sklearn import cross_validation

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


def test_get_set_train_test_directly():
    X, Y = get_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

    mc = copper.ModelComparison()
    mc.X_train = X_train
    mc.y_train = y_train
    mc.X_test = X_test
    mc.y_test = y_test

    eq_(mc.X_train.shape, (150 * 0.8, 4))
    eq_(mc.y_train.shape, (150 * 0.8, ))
    eq_(mc.X_test.shape, (150 * 0.2, 4))
    eq_(mc.y_test.shape, (150 * 0.2, ))
    eq_(mc.X_train, X_train)
    eq_(mc.y_train, y_train)
    eq_(mc.X_test, X_test)
    eq_(mc.y_test, y_test)


def test_get_set_train_test_dataset():
    X, Y = get_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.6)
    train = np.hstack((X_train, y_train[np.newaxis].T))
    test = np.hstack((X_test, y_test[np.newaxis].T))
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    train = copper.Dataset(train)
    train.role[4] = train.TARGET
    test = copper.Dataset(test)
    test.role[4] = test.TARGET

    mc = copper.ModelComparison()
    mc.train = train
    mc.test = test

    eq_(mc.X_train.shape, (150 * 0.4, 4))
    eq_(mc.y_train.shape, (150 * 0.4, ))
    eq_(mc.X_test.shape, (150 * 0.6, 4))
    eq_(mc.y_test.shape, (150 * 0.6, ))
    eq_(mc.X_train, X_train)
    eq_(mc.y_train, y_train)
    eq_(mc.X_test, X_test)
    eq_(mc.y_test, y_test)


def test_train_test_split_iris():
    mc = copper.ModelComparison()
    mc.train_test_split(get_iris_ds(), test_size=0.4)
    eq_(mc.X_train.shape, (150 * 0.6, 4))
    eq_(mc.y_train.shape, (150 * 0.6, ))
    eq_(mc.X_test.shape, (150 * 0.4, 4))
    eq_(mc.y_test.shape, (150 * 0.4, ))
    eq_((mc.X_train, mc.y_train), mc.train)
    eq_((mc.X_test, mc.y_test), mc.test)


def test_get_set_algorithms():
    mc = copper.ModelComparison()
    mc.train_test_split(get_iris_ds(), test_size=0.4)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    mc['LR'] = lr
    eq_(mc['LR'], lr)

    lr2 = LogisticRegression(penalty='l1')
    mc['LR l1'] = lr2
    eq_(mc['LR l1'], lr2)
    eq_(len(mc), 2)


def test_del_algorithm():
    mc = copper.ModelComparison()
    mc.train_test_split(get_iris_ds(), test_size=0.4)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    mc['LR'] = lr
    eq_(mc['LR'], lr)

    lr2 = LogisticRegression(penalty='l1')
    mc['LR l1'] = lr2
    eq_(mc['LR l1'], lr2)
    eq_(len(mc), 2)

    del mc['LR']
    eq_(mc['LR l1'], lr2)
    eq_(len(mc), 1)

    del mc['LR l1']
    eq_(len(mc), 0)


@raises(Exception)
def test_del_algorithm_raise():
    mc = copper.ModelComparison()
    mc.sample(get_iris_ds(), test_size=0.4)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    mc['LR'] = lr
    eq_(mc['LR'], lr)

    lr2 = LogisticRegression(penalty='l1')
    mc['LR l1'] = lr2
    eq_(mc['LR l1'], lr2)

    del mc['LR']
    mc['LR']


@raises(Exception)
def test_no_auto_fit():
    mc = copper.ModelComparison()
    mc.sample(get_iris_ds(), test_size=0.4)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    mc['LR'] = lr

    mc['LR'].coef_  # Doesn't exist yet


def test_fit():
    mc = copper.ModelComparison()
    mc.train_test_split(get_iris_ds(), test_size=0.4)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr2 = LogisticRegression(penalty='l1')
    mc['LR'] = lr
    mc['LR l1'] = lr2

    mc.fit()
    ok_(mc['LR'].coef_ is not None)
    ok_(mc['LR l1'].coef_ is not None)
    ok_(mc['LR'] != mc['LR l1'])

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
