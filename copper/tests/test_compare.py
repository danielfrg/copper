from __future__ import division
import math
import random
import copper
import numpy as np
import pandas as pd
from nose.tools import raises, ok_
from copper.tests.utils import eq_
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

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

# -----------------------------------------------------------------------------
#                           Train and test values

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


def test_get_set_train_test_dataset_property():
    X, Y = get_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.6)

    train = np.hstack((X_train, y_train[np.newaxis].T))
    train = pd.DataFrame(train)
    train = copper.Dataset(train)
    train.role[4] = train.TARGET

    test = np.hstack((X_test, y_test[np.newaxis].T))
    test = pd.DataFrame(test)
    test = copper.Dataset(test)
    test.role[4] = test.TARGET
    # --
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


def test_train_test_split():
    ds = get_iris_ds()
    mc = copper.ModelComparison()
    state = int(math.floor(random.random() * 1000))
    mc.train_test_split(ds, test_size=0.4, random_state=state)
    eq_(mc.X_train.shape, (150 * 0.6, 4))
    eq_(mc.y_train.shape, (150 * 0.6, ))
    eq_(mc.X_test.shape, (150 * 0.4, 4))
    eq_(mc.y_test.shape, (150 * 0.4, ))
    eq_((mc.X_train, mc.y_train), mc.train)
    eq_((mc.X_test, mc.y_test), mc.test)
    eq_(mc.le, None)
    # --
    X, Y = get_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=state)
    eq_(mc.X_train, X_train)
    eq_(mc.y_train, y_train)
    eq_(mc.X_test, X_test)
    eq_(mc.y_test, y_test)


# -----------------------------------------------------------------------------
#                        Algorithms list/dictionary

def test_get_set_algorithms():
    mc = copper.ModelComparison()
    lr = LogisticRegression()
    mc['LR'] = lr
    eq_(mc['LR'], lr)

    lr2 = LogisticRegression(penalty='l1')
    mc['LR l1'] = lr2
    eq_(mc['LR l1'], lr2)
    eq_(len(mc), 2)


def test_del_algorithm_len():
    mc = copper.ModelComparison()
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


@raises(KeyError)
def test_deleted_algorithm():
    mc = copper.ModelComparison()
    lr = LogisticRegression()
    mc['LR'] = lr
    eq_(mc['LR'], lr)

    lr2 = LogisticRegression(penalty='l1')
    mc['LR l1'] = lr2
    eq_(mc['LR l1'], lr2)

    del mc['LR']
    eq_(mc['LR l1'], lr2)  # Not deleted
    mc['LR']  # deleted

# -----------------------------------------------------------------------------


@raises(AttributeError)
def test_no_auto_fit():
    mc = copper.ModelComparison()
    lr = LogisticRegression()
    mc['LR'] = lr

    mc['LR'].coef_  # Doesn't exist yet


def test_fit():
    ds = get_iris_ds()
    mc = copper.ModelComparison()
    mc.train_test_split(ds, test_size=0.4)


    lr = LogisticRegression()
    lr2 = LogisticRegression(penalty='l1')
    mc['LR'] = lr
    mc['LR l1'] = lr2

    mc.fit()
    ok_(mc['LR'].coef_ is not None)
    ok_(mc['LR l1'].coef_ is not None)
    ok_(mc['LR'] != mc['LR l1'])


# -----------------------------------------------------------------------------
#                        With target as string

def get_iris_ds_string():
    ds = get_iris_ds()
    ds.type['Target'] = ds.CATEGORY
    ds['Target'] = ds['Target'].apply(lambda x: str(x))
    ds['Target'][ds['Target'] == '0'] = 'Iris-A'
    ds['Target'][ds['Target'] == '1'] = 'Iris-B'
    ds['Target'][ds['Target'] == '2'] = 'Iris-C'
    eq_(ds.metadata['dtype']['Target'], object)
    return ds


def test_train_test_split_string():
    ds = get_iris_ds_string()
    mc = copper.ModelComparison()
    state = int(math.floor(random.random() * 1000))
    mc.train_test_split(ds, test_size=0.4, random_state=state)
    eq_(mc.X_train.shape, (150 * 0.6, 4))
    eq_(mc.y_train.shape, (150 * 0.6, ))
    eq_(mc.X_test.shape, (150 * 0.4, 4))
    eq_(mc.y_test.shape, (150 * 0.4, ))
    eq_((mc.X_train, mc.y_train), mc.train)
    eq_((mc.X_test, mc.y_test), mc.test)
    eq_(mc.le.classes_.tolist(), ['Iris-A', 'Iris-B', 'Iris-C'])
    # --
    X, Y = get_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=state)
    eq_(mc.X_train, X_train)
    eq_(mc.y_train, y_train)
    eq_(mc.X_test, X_test)
    eq_(mc.y_test, y_test)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vs', '--nologcapture'], exit=False)
